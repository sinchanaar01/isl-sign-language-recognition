"""Validate external clip metadata/media and make a signer-independent manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from src.isl_recognition.preprocessing import signer_independent_split

REQUIRED_COLUMNS = {"video_path", "label", "signer_id"}


def video_is_readable(path: Path) -> bool:
    """Decode at least one frame, failing if OpenCV cannot inspect this file."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for the corrupt-video check. Install app dependencies."
        ) from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size)
    finally:
        capture.release()


def make_manifest(
    clips_csv: Path,
    output_csv: Path = Path("data/processed/manifest.csv"),
    report_path: Path = Path("reports/data_manifest.json"),
    seed: int = 42,
    video_probe: Callable[[Path], bool] = video_is_readable,
) -> dict:
    """Validate all rows, split by signer, and save counts plus the manifest."""
    clips_csv = Path(clips_csv)
    if not clips_csv.is_file():
        raise FileNotFoundError(
            f"Licensed external clip CSV not found: {clips_csv}. Prepare "
            "`data/raw/downloaded/clips.csv` with documented signer IDs; keep "
            "`data/raw/own/clips.csv` reserved for held-out live-demo tests."
        )
    with clips_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"{clips_csv} must include columns {sorted(REQUIRED_COLUMNS)}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(
            f"{clips_csv} has no clips. Record labeled clips before manifest creation."
        )

    problems: list[str] = []
    valid: list[dict[str, str]] = []
    for line_number, row in enumerate(rows, start=2):
        label = (row.get("label") or "").strip()
        signer = (row.get("signer_id") or "").strip()
        raw_path = (row.get("video_path") or "").strip()
        if not label or not signer or not raw_path:
            problems.append(
                f"line {line_number}: blank video_path, label, or signer_id"
            )
            continue
        video_path = Path(raw_path)
        if not video_path.is_absolute():
            video_path = clips_csv.parent / video_path
        video_path = video_path.resolve()
        if not video_path.is_file():
            problems.append(f"line {line_number}: missing video file: {video_path}")
            continue
        if not video_probe(video_path):
            problems.append(
                f"line {line_number}: corrupt/unreadable video: {video_path}"
            )
            continue
        valid.append(
            {
                "video_path": str(video_path),
                "label": label,
                "signer_id": signer,
            }
        )

    signers = sorted({row["signer_id"] for row in valid})
    if len(signers) < 3:
        problems.append(
            f"Only {len(signers)} distinct readable signers found; at least 3 are "
            "required for train/val/test without signer leakage."
        )
    if problems:
        raise ValueError(
            "Manifest validation failed; fix the listed rows and rerun:\n- "
            + "\n- ".join(problems)
        )

    splits = signer_independent_split([row["signer_id"] for row in valid], seed=seed)
    for row, split in zip(valid, splits):
        row["split"] = str(split)

    class_splits: dict[str, set[str]] = defaultdict(set)
    split_signers: dict[str, set[str]] = defaultdict(set)
    for row in valid:
        class_splits[row["label"]].add(row["split"])
        split_signers[row["split"]].add(row["signer_id"])
    missing_training_labels = sorted(
        label
        for label, assigned_splits in class_splits.items()
        if "train" not in assigned_splits
    )
    if missing_training_labels:
        raise ValueError(
            "These classes have no samples from a training signer: "
            f"{missing_training_labels}. Collect each sign from signers assigned "
            "to train, or adjust the split seed after inspecting signer/class coverage."
        )
    missing_partitions = {"train", "val", "test"} - {row["split"] for row in valid}
    if missing_partitions:
        raise ValueError(
            f"Signer grouping left empty splits {sorted(missing_partitions)}; "
            "collect clips from more balanced signer groups."
        )

    class_counts: dict[str, dict[str, int]] = {}
    for label in sorted(class_splits):
        class_counts[label] = dict(
            Counter(row["split"] for row in valid if row["label"] == label)
        )
    summary = {
        "source_csv": str(clips_csv.resolve()),
        "seed": seed,
        "clip_count": len(valid),
        "signer_count": len(signers),
        "class_count": len(class_counts),
        "split_clip_counts": dict(Counter(row["split"] for row in valid)),
        "split_signer_counts": {
            split: len(split_signers[split]) for split in ("train", "val", "test")
        },
        "class_counts_by_split": class_counts,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("video_path", "label", "signer_id", "split")
        )
        writer.writeheader()
        writer.writerows(valid)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {output_csv}; counts: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips", type=Path, default=Path("data/raw/downloaded/clips.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/manifest.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data_manifest.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_manifest(args.clips, args.output, args.report, args.seed)


if __name__ == "__main__":
    main()
