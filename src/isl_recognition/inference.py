"""Prediction smoothing and checkpoint loading helpers."""

from __future__ import annotations

from collections import deque
from typing import Any


class PredictionSmoother:
    """Average recent per-class probabilities and report the winning prediction."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._values: deque[dict[str, float]] = deque(maxlen=window_size)

    def update(self, probabilities: dict[str, float]) -> tuple[str, float]:
        if not probabilities:
            raise ValueError("probabilities cannot be empty")
        if any(not 0 <= value <= 1 for value in probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        self._values.append(probabilities)
        labels = {label for value in self._values for label in value}
        averaged = {
            label: sum(value.get(label, 0.0) for value in self._values) / len(self._values)
            for label in labels
        }
        label = max(averaged, key=averaged.__getitem__)
        return label, averaged[label]


def load_landmark_model(checkpoint_path: str) -> tuple[Any, list[str], int]:
    """Load a trained PyTorch landmark model; imports torch only when requested."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the 'app' dependencies to load model checkpoints") from exc
    from .models import build_landmark_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    class_to_idx = checkpoint["class_to_idx"]
    model = build_landmark_model(
        architecture,
        num_classes=len(class_to_idx),
        input_size=checkpoint["input_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    classes = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    return model, classes, checkpoint["sequence_length"]

