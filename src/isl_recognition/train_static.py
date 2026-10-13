"""Train three-seed MobileNetV2 baseline on signer-separated static images."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import measure_model_latency, save_evaluation_artifacts
from .models import build_mobilenet_v2


def validate_signer_separation(*datasets: Any) -> dict[str, set[str]]:
    """Require filenames named SIGNER_ID__sample.ext and reject split leakage."""
    split_signers: dict[str, set[str]] = {}
    for split, dataset in zip(("train", "val", "test"), datasets):
        if not dataset.samples:
            raise ValueError(f"{split} image split is empty")
        signers: set[str] = set()
        for filename, _ in dataset.samples:
            stem = Path(filename).stem
            if "__" not in stem:
                raise ValueError(
                    f"Image {filename} needs signer-prefixed filename "
                    "`SIGNER_ID__sample.ext` to verify a signer-independent split."
                )
            signer = stem.split("__", maxsplit=1)[0]
            if not signer:
                raise ValueError(f"Image {filename} has an empty signer ID prefix")
            signers.add(signer)
        split_signers[split] = signers
    names = tuple(split_signers)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = split_signers[first] & split_signers[second]
            if overlap:
                raise ValueError(
                    f"Signer leakage between {first} and {second}: {sorted(overlap)}"
                )
    return split_signers


def train_static(
    data_directory: str | Path = "data/raw/static",
    output_directory: str | Path = "models",
    reports_directory: str | Path = "reports",
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one deterministic MobileNetV2 run."""
    import mlflow
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_directory)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_data = datasets.ImageFolder(root / "train", transform=train_transform)
    val_data = datasets.ImageFolder(root / "val", transform=evaluation_transform)
    test_data = datasets.ImageFolder(root / "test", transform=evaluation_transform)
    class_names = train_data.classes
    if len(class_names) < 2:
        raise ValueError("Static image training requires at least two class folders")
    if any(
        dataset.class_to_idx != train_data.class_to_idx
        for dataset in (val_data, test_data)
    ):
        raise ValueError(
            "Train, val, and test folders must contain exactly the same class folders"
        )
    split_signers = validate_signer_separation(train_data, val_data, test_data)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = build_mobilenet_v2(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_directory)
    reports.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    checkpoint_path = output / f"mobilenet_v2_seed{seed}.pt"
    metrics_prefix = f"mobilenet_v2_seed{seed}"
    best_accuracy = -1.0
    mlflow.set_experiment("isl-static-model-comparison")
    with mlflow.start_run(run_name=metrics_prefix):
        mlflow.log_params({
            "architecture": "mobilenet_v2",
            "input_size": 224,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "signer_count": len(set.union(*split_signers.values())),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        })
        for epoch in range(epochs):
            model.train()
            train_loss = train_correct = train_total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += len(labels)

            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
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
                f"{metrics_prefix} epoch {epoch + 1}/{epochs}: "
                f"train_acc={train_correct / train_total:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "architecture": "mobilenet_v2",
                        "state_dict": model.state_dict(),
                        "class_to_idx": train_data.class_to_idx,
                        "input_size": 224,
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
        sample = None
        with torch.inference_mode():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                sample = inputs[:1] if sample is None else sample
                predictions.extend(model(inputs).argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())
        latency = measure_model_latency(model, sample, device)
        metrics = save_evaluation_artifacts(
            targets, predictions, class_names, reports, metrics_prefix, latency
        )
        run_record = {
            "run_id": run_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed_real_training_run",
            "architecture": "mobilenet_v2",
            "seed": seed,
            "device": str(device),
            "class_names": class_names,
            "signer_count": len(set.union(*split_signers.values())),
            "split_signer_counts": {
                split: len(signers) for split, signers in split_signers.items()
            },
            "samples": {
                "train": len(train_data), "val": len(val_data), "test": len(test_data)
            },
            "best_validation_accuracy": best_accuracy,
            "test_metrics": metrics,
            "model_only_latency_ms": latency,
            "checkpoint": str(checkpoint_path),
        }
        run_path = reports / "runs" / f"{run_id}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["per_class"]["macro avg"]["f1-score"],
            "model_only_latency_ms": latency,
        })
        for artifact in (
            checkpoint_path,
            reports / f"{metrics_prefix}_metrics.json",
            reports / f"{metrics_prefix}_confusion_matrix.png",
            reports / f"{metrics_prefix}_error_analysis.md",
            run_path,
        ):
            mlflow.log_artifact(str(artifact))
        print(
            f"{metrics_prefix}: test_accuracy={metrics['accuracy']:.4f}; "
            f"macro_f1={metrics['per_class']['macro avg']['f1-score']:.4f}; "
            f"model_latency_ms={latency:.2f}"
        )
        return run_record


def train_static_seeds(
    data_directory: str | Path,
    seeds: list[int],
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least three distinct seeds")
    runs = [
        train_static(data_directory, epochs=epochs, batch_size=batch_size, seed=seed)
        for seed in seeds
    ]
    accuracies = [run["test_metrics"]["accuracy"] for run in runs]
    macro_f1 = [
        run["test_metrics"]["per_class"]["macro avg"]["f1-score"] for run in runs
    ]
    summary = {
        "kind": "real_training_runs_only",
        "run_ids": [run["run_id"] for run in runs],
        "seeds": seeds,
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_population_std": statistics.pstdev(accuracies),
        "macro_f1_mean": statistics.mean(macro_f1),
        "macro_f1_population_std": statistics.pstdev(macro_f1),
    }
    path = Path("reports/runs/mobilenet_v2_seed_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/static")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    summary = train_static_seeds(
        args.data_dir, args.seeds, args.epochs, args.batch_size
    )
    Path("reports/runs").mkdir(parents=True, exist_ok=True)
    Path("reports/runs/mobilenet_v2_seed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
