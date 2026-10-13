"""Prediction smoothing and checkpoint loading helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class VideoInferenceResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_only_latency_ms: float
    end_to_end_latency_ms: float
    extracted_frame_count: int


def load_landmark_model(
    checkpoint_path: str,
) -> tuple[Any, list[str], int, str]:
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
    return (
        model,
        classes,
        checkpoint["sequence_length"],
        checkpoint.get("modality", "hands_pose"),
    )


def predict_video_file(
    video_path: str | Path,
    model: Any,
    classes: list[str],
    sequence_length: int,
    modality: str,
) -> VideoInferenceResult:
    """Run MediaPipe extraction and trained model inference on one video clip."""
    import time

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch before running video inference") from exc

    from .data import extract_video_landmarks
    from .evaluation import measure_model_latency
    from .preprocessing import normalize_sequence, select_modality

    path = Path(video_path)
    started = time.perf_counter()
    frames = extract_video_landmarks(path)
    sequence = select_modality(normalize_sequence(frames, sequence_length), modality)
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Loaded model has no parameters") from exc
    sample = torch.from_numpy(sequence).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(sample)
        probabilities_tensor = torch.softmax(logits[0], dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_to_end_ms = (time.perf_counter() - started) * 1000
    model_only_ms = measure_model_latency(model, sample, device)
    probabilities = probabilities_tensor.detach().cpu().tolist()
    class_probabilities = {
        name: float(probability)
        for name, probability in zip(classes, probabilities)
    }
    label, confidence = max(
        class_probabilities.items(), key=lambda item: item[1]
    )
    return VideoInferenceResult(
        label=label,
        confidence=confidence,
        probabilities=class_probabilities,
        model_only_latency_ms=model_only_ms,
        end_to_end_latency_ms=end_to_end_ms,
        extracted_frame_count=len(frames),
    )
