"""Create a video-label CSV from locally acquired WLASL ASL videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def create_manifest(
    annotations_path: str | Path,
    video_directory: str | Path,
    output_csv: str | Path,
    max_classes: int = 30,
) -> Path:
    if max_classes < 2:
        raise ValueError("max_classes must be at least 2")
    annotations_path = Path(annotations_path)
    video_directory = Path(video_directory)
    output_csv = Path(output_csv)
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    found: list[tuple[str, str, str]] = []
    missing_signer = 0
    for category in annotations:
        label = category["gloss"]
        for instance in category.get("instances", []):
            video_id = str(instance["video_id"])
            video_path = video_directory / f"{video_id}.mp4"
            if not video_path.is_file():
                continue
            signer_id = instance.get("signer_id")
            if signer_id is None or str(signer_id).strip() == "":
                missing_signer += 1
                continue
            found.append((label, str(signer_id), str(video_path.resolve())))
    if not found:
        raise FileNotFoundError(
            f"No WLASL videos matching annotation IDs were found in {video_directory}"
        )
    counts = Counter(label for label, _, _ in found)
    selected = {label for label, _ in counts.most_common(max_classes)}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "label", "signer_id"])
        writer.writerows(
            (video_path, label, signer)
            for label, signer, video_path in found
            if label in selected
        )
    print(
        f"Wrote {sum(counts[label] for label in selected)} local ASL clips across "
        f"{len(selected)} glosses to {output_csv}. "
        f"Skipped {missing_signer} clips without signer IDs."
    )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/raw/WLASL_v0.3.json")
    parser.add_argument("--videos", default="data/raw/wlasl/videos")
    parser.add_argument("--output", default="data/raw/wlasl_manifest.csv")
    parser.add_argument("--max-classes", type=int, default=30)
    args = parser.parse_args()
    create_manifest(args.annotations, args.videos, args.output, args.max_classes)


if __name__ == "__main__":
    main()
