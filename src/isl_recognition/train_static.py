"""Fine-tune MobileNetV2 for static sign images stored in ImageFolder layout."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .evaluation import measure_model_latency, save_evaluation_artifacts
from .models import build_mobilenet_v2


def train_static(
    data_directory: str | Path = "data/raw/static",
    output_directory: str | Path = "models",
    reports_directory: str | Path = "reports",
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
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
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_data = datasets.ImageFolder(root / "train", transform=train_transform)
    val_data = datasets.ImageFolder(root / "val", transform=test_transform)
    test_data = datasets.ImageFolder(root / "test", transform=test_transform)
    class_names = train_data.classes
    if len(class_names) < 2:
        raise ValueError("Static image training requires at least two class folders")
    if not train_data or not val_data or not test_data:
        raise ValueError("Train, val, and test image folders must each contain images")
    if any(dataset.class_to_idx != train_data.class_to_idx for dataset in (val_data, test_data)):
        raise ValueError("Train, val, and test folders must contain the same class directories")
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
    checkpoint_path = output / "mobilenet_v2_static.pt"
    best_accuracy = -1.0
    mlflow.set_experiment("isl-static-model-comparison")
    with mlflow.start_run(run_name="mobilenet_v2"):
        mlflow.log_params({
            "architecture": "mobilenet_v2",
            "input_size": 224,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
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
                f"MobileNetV2 epoch {epoch + 1}/{epochs}: "
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
            targets, predictions, class_names, reports_directory, "mobilenet_v2", latency
        )
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_mean_inference_latency_ms": latency,
        })
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(Path(reports_directory) / "mobilenet_v2_metrics.json"))
        mlflow.log_artifact(str(Path(reports_directory) / "mobilenet_v2_confusion_matrix.png"))
        print(
            f"MobileNetV2 held-out test accuracy={metrics['accuracy']:.4f}; "
            f"mean latency={latency:.2f} ms; checkpoint={checkpoint_path}"
        )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/static")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_static(
        args.data_dir, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/mobilenet_v2_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
