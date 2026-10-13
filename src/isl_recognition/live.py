"""Primary local webcam demo with motion-gated isolated-sign inference."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agent import Prediction, SentenceAgent
from .inference import load_landmark_model
from .preprocessing import (
    FEATURES_PER_FRAME,
    normalize_frame,
    normalize_sequence,
    select_modality,
)
from .sign_activity import SignActivityDetector, SignSegment

LIVE_SEQUENCE_LENGTH = 32


def landmark_features(
    results: Any,
) -> tuple[NDArray[np.float32], list[tuple[float, float]]]:
    """Convert Holistic results to normalized model features and wrist positions."""
    values = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    if results.pose_landmarks:
        for index, point in enumerate(results.pose_landmarks.landmark):
            values[index * 3 : index * 3 + 3] = (point.x, point.y, point.z)
            values[99 + index] = point.visibility

    wrists: list[tuple[float, float]] = []
    for hand_landmarks, start in (
        (results.left_hand_landmarks, 132),
        (results.right_hand_landmarks, 195),
    ):
        if hand_landmarks:
            for index, point in enumerate(hand_landmarks.landmark):
                values[start + index * 3 : start + index * 3 + 3] = (
                    point.x,
                    point.y,
                    point.z,
                )
            wrist = hand_landmarks.landmark[0]
            wrists.append((float(wrist.x), float(wrist.y)))
    return normalize_frame(values), wrists


def predict_segment(
    model: Any, segment: SignSegment, classes: list[str], modality: str
) -> tuple[str, float]:
    """Predict exactly once from a completed segment resampled to 32 frames."""
    import torch

    sequence = normalize_sequence(np.stack(segment.frames), LIVE_SEQUENCE_LENGTH)
    sequence = select_modality(sequence, modality)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor)[0], dim=0).cpu().tolist()
    return max(
        zip(classes, probabilities, strict=True),
        key=lambda item: item[1],
    )


def _save_run_record(
    destination: Path,
    camera_index: int,
    device: str,
    frame_count: int,
    elapsed_seconds: float,
    displayed_word_latencies_ms: list[float],
    prediction_count: int,
) -> dict[str, Any]:
    fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else None
    record = {
        "kind": "measured_live_camera_run",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None,
        "camera_index": camera_index,
        "device": device,
        "captured_frame_count": frame_count,
        "elapsed_seconds": elapsed_seconds,
        "measured_processing_fps": fps,
        "displayed_sign_count": prediction_count,
        "camera_frame_to_displayed_word_latency_ms": {
            "measurement_count": len(displayed_word_latencies_ms),
            "mean": (
                statistics.mean(displayed_word_latencies_ms)
                if displayed_word_latencies_ms
                else None
            ),
            "median": (
                statistics.median(displayed_word_latencies_ms)
                if displayed_word_latencies_ms
                else None
            ),
            "observations": displayed_word_latencies_ms,
        },
        "measurement_scope": (
            "Local capture-read timestamp through OpenCV window update and waitKey; "
            "does not include physical display scanout."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = destination / f"live_camera_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["run_record"] = str(path)
    return record


def run_camera(
    model_path: Path,
    camera_index: int = 0,
    motion_threshold: float = 0.02,
    onset_frames: int = 3,
    end_frames: int = 12,
    cooldown_seconds: float = 0.8,
) -> int:
    """Run the local webcam UI and save only measurements observed in that run."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {model_path}. Acquire licensed/consented ISL "
            "data, train a real model, then run `python -m "
            "src.isl_recognition.train_landmarks` before starting live mode."
        )
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Live mode requires the app dependencies. Install with "
            '`python -m pip install -e ".[app]"`.'
        ) from exc

    model, classes, _, modality = load_landmark_model(str(model_path))
    try:
        device = str(next(model.parameters()).device)
    except StopIteration as exc:
        raise RuntimeError("The loaded checkpoint model has no parameters.") from exc
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise OSError(
            f"Could not open camera {camera_index}. Close other camera apps or pass "
            "--camera-index."
        )

    detector = SignActivityDetector(
        motion_threshold=motion_threshold,
        onset_frames=onset_frames,
        end_frames=end_frames,
        cooldown_seconds=cooldown_seconds,
    )
    sentence_agent = SentenceAgent(api_key="")
    sign_buffer: list[Prediction] = []
    latest_word = "—"
    latest_confidence = 0.0
    sentence = "Ready to sign"
    latencies: list[float] = []
    pending_display_start: float | None = None
    frame_count = 0
    prediction_count = 0
    started = time.perf_counter()
    last_elapsed = 0.0
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
    try:
        while True:
            ok, image = capture.read()
            captured_at = time.perf_counter()
            if not ok:
                raise OSError("Camera stopped returning frames.")
            frame_count += 1
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            feature_frame, wrists = landmark_features(results)
            update = detector.update(
                feature_frame,
                wrists,
                hands_visible=bool(wrists),
                timestamp=captured_at,
            )
            if update.segment is not None:
                cv2.putText(
                    image,
                    "State: processing",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )
                cv2.imshow("ISL live recognition - press q to quit", image)
                cv2.waitKey(1)
                latest_word, latest_confidence = predict_segment(
                    model, update.segment, classes, modality
                )
                sign_buffer.append(Prediction(latest_word, latest_confidence))
                sentence = sentence_agent.generate(sign_buffer).text
                prediction_count += 1
                pending_display_start = update.segment.started_at
                detector.finish_processing()

            drawing = mp.solutions.drawing_utils
            holistic_connections = mp.solutions.holistic
            drawing.draw_landmarks(
                image, results.pose_landmarks, holistic_connections.POSE_CONNECTIONS
            )
            drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                holistic_connections.HAND_CONNECTIONS,
            )
            drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                holistic_connections.HAND_CONNECTIONS,
            )
            last_elapsed = captured_at - started
            fps = frame_count / last_elapsed if last_elapsed > 0 else 0.0
            overlay = (
                f"State: {detector.state} | FPS: {fps:.1f}",
                f"Word: {latest_word} | Confidence: {latest_confidence:.0%}",
                f"Sentence: {sentence}",
            )
            for index, text in enumerate(overlay):
                cv2.putText(
                    image,
                    text,
                    (12, 30 + index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (20, 220, 20),
                    2,
                )
            cv2.imshow("ISL live recognition - press q to quit", image)
            key = cv2.waitKey(1) & 0xFF
            displayed_at = time.perf_counter()
            if pending_display_start is not None:
                latencies.append((displayed_at - pending_display_start) * 1000)
                pending_display_start = None
            if key == ord("q"):
                break
    finally:
        capture.release()
        holistic.close()
        cv2.destroyAllWindows()

    record = _save_run_record(
        Path("reports/runs"),
        camera_index,
        device,
        frame_count,
        last_elapsed,
        latencies,
        prediction_count,
    )
    print(json.dumps(record, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("ISL_MODEL_PATH", "models/best_landmark.pt")),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--onset-frames", type=int, default=3)
    parser.add_argument("--end-frames", type=int, default=12)
    parser.add_argument("--cooldown-seconds", type=float, default=0.8)
    args = parser.parse_args()
    try:
        raise SystemExit(
            run_camera(
                args.model,
                args.camera_index,
                args.motion_threshold,
                args.onset_frames,
                args.end_frames,
                args.cooldown_seconds,
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
