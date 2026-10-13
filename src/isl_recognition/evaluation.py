"""Classification metrics, confusion-matrix plot, and inference latency."""

from __future__ import annotations

import time
import json
from pathlib import Path


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn and matplotlib to evaluate models") from exc
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "misclassified_pairs": [
            {
                "actual": class_names[actual],
                "predicted": class_names[predicted],
                "count": int(count),
            }
            for actual, row in enumerate(matrix)
            for predicted, count in enumerate(row)
            if actual != predicted and count
        ],
    }


def save_evaluation_artifacts(
    y_true,
    y_pred,
    class_names: list[str],
    output_directory: str | Path,
    prefix: str,
    latency_ms: float,
) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to create evaluation plots") from exc
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["mean_inference_latency_ms"] = float(latency_ms)

    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(
        figsize=(max(7, len(class_names) * 0.45), max(6, len(class_names) * 0.4))
    )
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{prefix} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(output / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(figure)

    metrics_path = output / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confused = sorted(
        metrics["misclassified_pairs"],
        key=lambda item: item["count"],
        reverse=True,
    )[:5]
    if confused:
        lines = [
            f"# Error analysis: {prefix}",
            "",
            "Top class confusions measured on the held-out test predictions:",
            "",
            "| Actual | Predicted | Count |",
            "| --- | --- | ---: |",
        ]
        lines.extend(
            f"| {item['actual']} | {item['predicted']} | {item['count']} |"
            for item in confused
        )
        lines += [
            "",
            "Counts come directly from this run's confusion matrix. Inspect the clips "
            "for these pairs and determine whether errors reflect visual similarity, "
            "label ambiguity, or insufficient signer coverage before drawing causal "
            "conclusions.",
            "",
        ]
    else:
        lines = [
            f"# Error analysis: {prefix}",
            "",
            "No off-diagonal errors were observed in this held-out run. This does not "
            "establish broad generalization; inspect the sample size and signer "
            "coverage in the corresponding run record.",
            "",
        ]
    (output / f"{prefix}_error_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return metrics


def measure_model_latency(model, sample, device, repetitions: int = 100) -> float:
    import torch

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    model.eval()
    sample = sample.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(5):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / repetitions
