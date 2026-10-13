"""Train and compare LSTM, GRU, and Transformer landmark classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np

from .evaluation import measure_model_latency, save_evaluation_artifacts
from .models import build_landmark_model
from .preprocessing import FEATURES_PER_FRAME, augment_sequence, normalize_sequence


def train_architecture(
    manifest_path: str | Path,
    architecture: str,
    output_directory: str | Path = "models",
    reports_directory: str | Path = "reports",
    sequence_length: int = 32,
    epochs: int = 40,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    import mlflow
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if architecture not in {"lstm", "gru", "transformer"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = Path(manifest_path)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    required = {"landmarks_path", "label", "split"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"{manifest} must contain {sorted(required)} columns and data")
    class_names = sorted({row["label"] for row in records})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in records):
            raise ValueError(f"No samples assigned to '{split}'")
    for row in records:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {row['split']!r}")
    train_labels = {row["label"] for row in records if row["split"] == "train"}
    if train_labels != set(class_names):
        missing = sorted(set(class_names) - train_labels)
        raise ValueError(
            "Every class must have at least one training sample; signer split leaves "
            f"these classes absent from train: {missing}"
        )

    class LandmarkDataset(Dataset):
        def __init__(self, split: str, augment: bool) -> None:
            self.rows = [row for row in records if row["split"] == split]
            self.augment = augment
            self.rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            path = Path(row["landmarks_path"])
            if not path.is_absolute():
                path = manifest.parent / path
            frames = np.load(path, allow_pickle=False)
            sequence = normalize_sequence(frames, sequence_length)
            if self.augment:
                sequence = augment_sequence(sequence, self.rng)
            return torch.from_numpy(sequence), class_to_idx[row["label"]]

    train_data = LandmarkDataset("train", augment=True)
    val_data = LandmarkDataset("val", augment=False)
    test_data = LandmarkDataset("test", augment=False)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = build_landmark_model(architecture, len(class_names), FEATURES_PER_FRAME)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_directory)

    mlflow.set_experiment("isl-landmark-model-comparison")
    with mlflow.start_run(run_name=architecture):
        mlflow.log_params({
            "architecture": architecture,
            "sequence_length": sequence_length,
            "input_size": FEATURES_PER_FRAME,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "signers": len({row["signer_id"] for row in records if "signer_id" in row}),
        })
        checkpoint_path = output / f"{architecture}_landmark.pt"
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += len(labels)

            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.long)
                    logits = model(inputs)
                    val_loss += criterion(logits, labels).item() * len(labels)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += len(labels)
            val_accuracy = val_correct / val_total
            mlflow.log_metrics(
                {
                    "train_loss": train_loss / train_total,
                    "train_accuracy": train_correct / train_total,
                    "val_loss": val_loss / val_total,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            print(
                f"{architecture} epoch {epoch + 1}/{epochs}: "
                f"train_acc={train_correct / train_total:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "architecture": architecture,
                        "state_dict": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "input_size": FEATURES_PER_FRAME,
                        "sequence_length": sequence_length,
                        "seed": seed,
                        "best_val_accuracy": best_accuracy,
                    },
                    checkpoint_path,
                )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        targets: list[int] = []
        predictions: list[int] = []
        sample_input = None
        with torch.inference_mode():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                if sample_input is None:
                    sample_input = inputs[:1]
                logits = model(inputs)
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())
        latency = measure_model_latency(model, sample_input, device)
        metrics = save_evaluation_artifacts(
            targets, predictions, class_names, reports, architecture, latency
        )
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_mean_inference_latency_ms": latency,
        })
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(reports / f"{architecture}_metrics.json"))
        mlflow.log_artifact(str(reports / f"{architecture}_confusion_matrix.png"))
        print(
            f"{architecture} held-out test accuracy={metrics['accuracy']:.4f}; "
            f"mean latency={latency:.2f} ms; checkpoint={checkpoint_path}"
        )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--architectures", nargs="+", default=["lstm", "gru", "transformer"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = {}
    for architecture in args.architectures:
        summary[architecture] = train_architecture(
            args.manifest,
            architecture,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    if not summary:
        raise ValueError("Provide at least one architecture to train")
    selected = max(
        args.architectures,
        key=lambda name: torch_checkpoint_val_accuracy(
            Path("models") / f"{name}_landmark.pt"
        ),
    )
    shutil.copyfile(
        Path("models") / f"{selected}_landmark.pt",
        Path("models/best_landmark.pt"),
    )
    print(f"Selected {selected} by validation accuracy for webcam inference.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/landmark_model_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def torch_checkpoint_val_accuracy(checkpoint_path: Path) -> float:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return float(checkpoint["best_val_accuracy"])


if __name__ == "__main__":
    main()
