"""Video-to-landmark extraction and manifest generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .preprocessing import FEATURES_PER_FRAME, normalize_frame


def extract_video_landmarks(video_path: str | Path, max_frames: int = 96) -> np.ndarray:
    """Extract normalized Holistic pose/hand landmarks from one video."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'app' dependencies to extract video landmarks"
        ) from exc
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Video has no readable frames: {path}")
    selected_indices = np.linspace(
        0, total_frames - 1, min(total_frames, max_frames), dtype=np.int64
    )
    selected = set(selected_indices.tolist())
    frames: list[np.ndarray] = []
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            for frame_index in range(total_frames):
                ok, bgr = capture.read()
                if not ok:
                    break
                if frame_index not in selected:
                    continue
                results = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frame = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
                pose_offset = 0
                if results.pose_landmarks:
                    for index, landmark in enumerate(results.pose_landmarks.landmark):
                        frame[pose_offset + index * 3:pose_offset + index * 3 + 3] = (
                            landmark.x, landmark.y, landmark.z
                        )
                        frame[99 + index] = landmark.visibility
                hand_offset = 132
                for hand_landmarks, start in (
                    (results.left_hand_landmarks, hand_offset),
                    (results.right_hand_landmarks, hand_offset + 63),
                ):
                    if hand_landmarks:
                        for index, landmark in enumerate(hand_landmarks.landmark):
                            frame[start + index * 3:start + index * 3 + 3] = (
                                landmark.x, landmark.y, landmark.z
                            )
                frames.append(normalize_frame(frame))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames could be processed from {path}")
    return np.stack(frames)


def prepare_manifest(
    source_csv: str | Path,
    output_directory: str | Path,
    seed: int = 42,
    max_frames: int = 96,
) -> Path:
    """Extract every source clip, create a signer-exclusive split manifest."""
    from .preprocessing import signer_independent_split

    source = Path(source_csv)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    failures: list[str] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "label", "signer_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{source} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = source.parent / video_path
            if not row["label"].strip() or not row["signer_id"].strip():
                failures.append(f"row {row_number}: blank label or signer_id")
                continue
            try:
                sequence = extract_video_landmarks(video_path, max_frames)
                name = f"clip_{row_number:06d}.npy"
                np.save(output / name, sequence, allow_pickle=False)
                records.append({
                    "video_path": str(video_path.resolve()),
                    "landmarks_path": name,
                    "label": row["label"].strip(),
                    "signer_id": row["signer_id"].strip(),
                })
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f"row {row_number} ({video_path}): {exc}")

    if failures:
        (output / "preprocessing_errors.txt").write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
    if len(records) < 3:
        raise RuntimeError(
            f"Only {len(records)} clips were processed successfully; at least 3 are required"
        )
    splits = signer_independent_split(
        [record["signer_id"] for record in records], seed=seed
    )
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "video_path", "landmarks_path", "label", "signer_id", "split"
        ])
        writer.writeheader()
        for record, split in zip(records, splits):
            writer.writerow({**record, "split": split})
    print(
        f"Processed {len(records)} clips; {len(failures)} failed. "
        f"Manifest: {manifest}"
    )
    if failures:
        print(f"Review per-row errors: {output / 'preprocessing_errors.txt'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=96)
    args = parser.parse_args()
    prepare_manifest(args.source_csv, args.output_dir, args.seed, args.max_frames)


if __name__ == "__main__":
    main()
