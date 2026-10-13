"""Train reproducible landmark-model and ablation grids with MLflow tracking."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import measure_model_latency, save_evaluation_artifacts
from .models import build_landmark_model
from .preprocessing import (
    FEATURES_HANDS_ONLY,
    FEATURES_PER_FRAME,
    augment_sequence,
    normalize_sequence,
    select_modality,
)


ARCHITECTURES = ("lstm", "gru", "transformer", "tcn")
MODALITIES = ("hands", "hands_pose")


def train_architecture(
    manifest_path: str | Path,
    architecture: str,
    output_directory: str | Path = "models",
    reports_directory: str | Path = "reports",
    sequence_length: int = 32,
    epochs: int = 40,
    batch_size: int = 32,
    seed: int = 42,
    modality: str = "hands_pose",
    use_augmentation: bool = True,
) -> dict[str, Any]:
    """Fit one configuration and save its validation-best checkpoint and test log."""
    import mlflow
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    if sequence_length not in {16, 32, 48}:
        raise ValueError("sequence_length must be 16, 32, or 48")
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
    required = {"landmarks_path", "label", "split", "signer_id"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"{manifest} must contain {sorted(required)} columns and data")
    class_names = sorted({row["label"] for row in records})
    if len(class_names) < 2:
        raise ValueError("At least two classes are required")
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in records):
            raise ValueError(f"No samples assigned to {split!r}")
    if any(row["split"] not in {"train", "val", "test"} for row in records):
        raise ValueError("Manifest contains a split other than train/val/test")
    if {row["label"] for row in records if row["split"] == "train"} != set(class_names):
        missing = sorted(
            set(class_names)
            - {row["label"] for row in records if row["split"] == "train"}
        )
        raise ValueError(f"Classes missing from train: {missing}")
    split_signers: dict[str, set[str]] = {
        split: {row["signer_id"] for row in records if row["split"] == split}
        for split in ("train", "val", "test")
    }
    if (
        split_signers["train"] & split_signers["val"]
        or split_signers["train"] & split_signers["test"]
        or split_signers["val"] & split_signers["test"]
    ):
        raise ValueError("Signer leakage detected between manifest splits")

    class LandmarkDataset(Dataset):
        def __init__(self, split: str) -> None:
            self.rows = [row for row in records if row["split"] == split]
            self.augment = use_augmentation and split == "train"
            self.rng = np.random.default_rng(
                seed + {"train": 0, "val": 1, "test": 2}[split]
            )

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
            sequence = select_modality(sequence, modality)
            return torch.from_numpy(sequence), class_to_idx[row["label"]]

    train_data = LandmarkDataset("train")
    val_data = LandmarkDataset("val")
    test_data = LandmarkDataset("test")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    input_size = (
        FEATURES_HANDS_ONLY if modality == "hands" else FEATURES_PER_FRAME
    )
    model = build_landmark_model(architecture, len(class_names), input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    run_id = uuid.uuid4().hex
    config_slug = (
        f"{architecture}_{modality}_aug{int(use_augmentation)}_"
        f"len{sequence_length}_seed{seed}"
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_directory)
    reports.mkdir(parents=True, exist_ok=True)
    run_log_directory = reports / "runs"
    run_log_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{config_slug}.pt"

    mlflow.set_experiment("isl-landmark-model-comparison")
    with mlflow.start_run(run_name=config_slug):
        mlflow.log_params({
            "architecture": architecture,
            "sequence_length": sequence_length,
            "modality": modality,
            "input_size": input_size,
            "augmentation": use_augmentation,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "signer_count": len({row["signer_id"] for row in records}),
            "class_count": len(class_names),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        })
        for epoch in range(epochs):
            model.train()
            train_loss = train_correct = train_total = 0
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
            val_correct = val_total = 0
            val_loss = 0.0
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
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
                f"{config_slug} epoch {epoch + 1}/{epochs}: "
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
                        "input_size": input_size,
                        "sequence_length": sequence_length,
                        "modality": modality,
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
                predictions.extend(model(inputs).argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())
        latency = measure_model_latency(model, sample_input, device)
        prefix = config_slug
        metrics = save_evaluation_artifacts(
            targets, predictions, class_names, reports, prefix, latency
        )
        end_to_end = measure_video_pipeline_latency(
            next(
                (
                    row["video_path"] for row in records
                    if row["split"] == "test" and row.get("video_path")
                ),
                None,
            ),
            model,
            device,
            sequence_length,
            modality,
        )
        result = {
            "run_id": run_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "architecture": architecture,
            "modality": modality,
            "augmentation": use_augmentation,
            "sequence_length": sequence_length,
            "seed": seed,
            "device": str(device),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "signer_count": len({row["signer_id"] for row in records}),
            "samples": {
                split: len(dataset)
                for split, dataset in (
                    ("train", train_data), ("val", val_data), ("test", test_data)
                )
            },
            "best_validation_accuracy": best_accuracy,
            "test_metrics": metrics,
            "model_only_latency_ms": latency,
            "end_to_end_latency": end_to_end,
            "checkpoint": str(checkpoint_path),
            "status": "completed_real_training_run",
        }
        run_path = run_log_directory / f"{run_id}.json"
        run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["per_class"]["macro avg"]["f1-score"],
            "model_only_latency_ms": latency,
        })
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(reports / f"{prefix}_metrics.json"))
        mlflow.log_artifact(str(reports / f"{prefix}_confusion_matrix.png"))
        mlflow.log_artifact(str(reports / f"{prefix}_error_analysis.md"))
        mlflow.log_artifact(str(run_path))
        print(
            f"{config_slug}: test_accuracy={metrics['accuracy']:.4f}; "
            f"macro_f1={metrics['per_class']['macro avg']['f1-score']:.4f}; "
            f"model_latency_ms={latency:.2f}"
        )
        return result


def measure_video_pipeline_latency(
    video_path: str | None,
    model: Any,
    device: Any,
    sequence_length: int,
    modality: str,
) -> dict[str, Any]:
    """Measure a real clip through MediaPipe, normalization, and model inference."""
    if not video_path:
        return {
            "status": "not_measured",
            "reason": "test manifest has no original video_path",
        }
    import time
    import torch
    from .data import extract_video_landmarks

    started = time.perf_counter()
    frames = extract_video_landmarks(video_path)
    sequence = normalize_sequence(frames, sequence_length)
    sequence = select_modality(sequence, modality)
    input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "status": "measured_real_video",
        "video_path": str(Path(video_path).resolve()),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "output_class_index": int(logits.argmax(dim=1).item()),
    }


def run_grid(
    manifest: str,
    architectures: list[str],
    seeds: list[int],
    modalities: list[str],
    augmentations: list[bool],
    sequence_lengths: list[int],
    epochs: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run the configured full seed/model/modality/augmentation/length grid."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed list must not contain duplicates")
    if len(seeds) < 3:
        raise ValueError("Use at least three seeds for mean/std model comparison")
    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    for architecture in architectures:
        for modality in modalities:
            for augmentation in augmentations:
                for sequence_length in sequence_lengths:
                    for seed in seeds:
                        result = train_architecture(
                            manifest,
                            architecture,
                            sequence_length=sequence_length,
                            epochs=epochs,
                            batch_size=batch_size,
                            seed=seed,
                            modality=modality,
                            use_augmentation=augmentation,
                        )
                        results.append(result)
                        if (
                            best_result is None
                            or result["best_validation_accuracy"]
                            > best_result["best_validation_accuracy"]
                        ):
                            best_result = result
                            import torch

                            checkpoint = torch.load(
                                result["checkpoint"], map_location="cpu",
                                weights_only=True,
                            )
                            Path("models").mkdir(exist_ok=True)
                            torch.save(checkpoint, "models/best_landmark.pt")
                            Path("models/class_map.json").write_text(
                                json.dumps(checkpoint["class_to_idx"], indent=2) + "\n",
                                encoding="utf-8",
                            )
                            write_model_card(result)

    Path("reports/runs").mkdir(parents=True, exist_ok=True)
    summary = summarize_trials(results)
    summary_path = Path("reports/runs/landmark_experiment_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_ablation_table(summary, Path("reports/ablation_table.csv"))
    return results


def summarize_trials(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate real runs by configuration, retaining each seed's observations."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result["architecture"],
            result["modality"],
            result["augmentation"],
            result["sequence_length"],
        )
        grouped.setdefault(key, []).append(result)
    rows = []
    for key, trials in sorted(grouped.items()):
        accuracies = [item["test_metrics"]["accuracy"] for item in trials]
        macro_f1s = [
            item["test_metrics"]["per_class"]["macro avg"]["f1-score"]
            for item in trials
        ]
        rows.append({
            "architecture": key[0],
            "modality": key[1],
            "augmentation": key[2],
            "sequence_length": key[3],
            "seeds": [item["seed"] for item in trials],
            "accuracy_mean": statistics.mean(accuracies),
            "accuracy_population_std": statistics.pstdev(accuracies),
            "macro_f1_mean": statistics.mean(macro_f1s),
            "macro_f1_population_std": statistics.pstdev(macro_f1s),
            "run_ids": [item["run_id"] for item in trials],
        })
    return {
        "kind": "real_training_runs_only",
        "run_count": len(results),
        "configurations": rows,
    }


def write_ablation_table(summary: dict[str, Any], destination: Path) -> None:
    fields = (
        "architecture", "modality", "augmentation", "sequence_length",
        "accuracy_mean", "accuracy_population_std", "macro_f1_mean",
        "macro_f1_population_std", "seeds", "run_ids",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["configurations"]:
            writer.writerow({**row, "seeds": json.dumps(row["seeds"]),
                             "run_ids": json.dumps(row["run_ids"])})


def write_model_card(result: dict[str, Any]) -> None:
    content = f"""# Best landmark model card

## Intended use

Research prototype for classifying isolated sign clips from the recorded dataset. It is not a continuous sign-language translator or interpreter replacement.

## Training provenance

- Selection: highest recorded validation accuracy among the completed experiment grid.
- Architecture: `{result['architecture']}`
- Input modality: `{result['modality']}`
- Sequence length: `{result['sequence_length']}`
- Training augmentation: `{result['augmentation']}`
- Random seed: `{result['seed']}`
- Training/validation/test sample counts: `{result['samples']}`
- Signer groups: `{result['signer_count']}` total; use the per-run log for split identities/counts.
- Run record: `reports/runs/{result['run_id']}.json`
- Test accuracy and limitations: see the run record; do not compare across different splits.

## Data and licensing

Dataset source/license must be documented in `data/sources.md`. Do not redistribute weights unless the source data license permits derived model distribution. Remove identifying metadata and obtain participant consent before publication.

## Limitations

This model is tied to its recorded label map and signer population. It does not recognize unseen vocabulary, encode facial/non-manual grammar, or guarantee robust communication.
"""
    Path("models/MODEL_CARD.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES)
    )
    parser.add_argument(
        "--augmentations", nargs="+", choices=("on", "off"), default=["on", "off"]
    )
    parser.add_argument(
        "--sequence-lengths", nargs="+", type=int, default=[16, 32, 48]
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    unknown = set(args.architectures) - set(ARCHITECTURES)
    if unknown:
        raise ValueError(f"Unknown architectures: {sorted(unknown)}")
    run_grid(
        args.manifest,
        args.architectures,
        args.seeds,
        args.modalities,
        [choice == "on" for choice in args.augmentations],
        args.sequence_lengths,
        args.epochs,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
