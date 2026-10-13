"""Record consented webcam clips into a signer-labeled dataset.

Controls while recording: q stops the recording loop; it is never saved.
After recording: r reviews the most recently saved clip, d deletes it, q exits.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import TextIO

CSV_FIELDS = ("video_path", "label", "signer_id")


def safe_component(value: str, field_name: str) -> str:
    """Return a conservative path component or reject it."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"{field_name} must contain letters or numbers")
    return cleaned


def next_clip_number(directory: Path, label: str) -> int:
    matches = re.compile(rf"^{re.escape(label)}_(\d+)\.mp4$", re.IGNORECASE)
    numbers = [
        int(match.group(1))
        for path in directory.glob("*.mp4")
        if (match := matches.match(path.name))
    ]
    return max(numbers, default=0) + 1


def append_manifest(
    csv_path: Path, video_path: Path, label: str, signer_id: str
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "video_path": str(video_path.resolve()),
                "label": label,
                "signer_id": signer_id,
            }
        )


def remove_manifest_entry(csv_path: Path, video_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(CSV_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Cannot safely edit malformed manifest: {csv_path}")
        rows = [
            row
            for row in reader
            if Path(row["video_path"]).resolve() != video_path.resolve()
        ]
        fieldnames = reader.fieldnames
    temporary_path = csv_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)


def play_clip(video_path: Path, cv2_module) -> None:
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not review recorded video: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2_module.imshow("Review last clip (press any key to close)", frame)
            if cv2_module.waitKey(25) >= 0:
                break
    finally:
        capture.release()
        cv2_module.destroyWindow("Review last clip (press any key to close)")


def record(
    signer_id: str,
    output_root: Path = Path("data/raw/own"),
    manifest_path: Path = Path("data/raw/clips.csv"),
    camera_index: int = 0,
    duration_seconds: float = 2.5,
    countdown_seconds: int = 3,
    input_stream: TextIO | None = None,
) -> None:
    """Capture labelled clips; each successful recording is immediately reviewed."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install the project app dependencies first."
        ) from exc
    if duration_seconds < 2.0 or duration_seconds > 3.0:
        raise ValueError("duration_seconds must be between 2 and 3 seconds")
    if countdown_seconds < 0:
        raise ValueError("countdown_seconds cannot be negative")

    import sys

    input_stream = input_stream or sys.stdin
    signer = safe_component(signer_id, "signer_id")
    signer_dir = output_root / signer
    signer_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise OSError(
            f"Could not open webcam index {camera_index}; close other camera apps "
            "or choose --camera-index."
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    latest: tuple[Path, str] | None = None
    print(
        "Dataset consent reminder: record only with informed consent; do not record "
        "bystanders. Label and review every clip. Enter a label (or q to stop)."
    )
    try:
        while True:
            label_input = input("Sign label: ").strip()
            if label_input.casefold() == "q":
                break
            try:
                label = safe_component(label_input, "label")
            except ValueError as exc:
                print(exc)
                continue
            clip_number = next_clip_number(signer_dir, label)
            video_path = signer_dir / f"{label}_{clip_number:03d}.mp4"

            interrupted = False
            for remaining in range(countdown_seconds, 0, -1):
                ok, frame = capture.read()
                if not ok:
                    raise OSError("Webcam stopped returning frames during countdown")
                cv2.putText(
                    frame,
                    f"{label}: recording in {remaining}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("ISL clip recorder", frame)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    interrupted = True
                    break
            if interrupted:
                break

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise OSError(
                    "Could not initialize MP4 writer; verify OpenCV codec support."
                )
            start_time = time.monotonic()
            frame_count = 0
            try:
                while time.monotonic() - start_time < duration_seconds:
                    ok, frame = capture.read()
                    if not ok:
                        raise OSError("Webcam stopped returning frames while recording")
                    cv2.putText(
                        frame,
                        f"RECORDING: {label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    writer.write(frame)
                    frame_count += 1
                    cv2.imshow("ISL clip recorder", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        interrupted = True
                        break
            finally:
                writer.release()
            if interrupted or frame_count == 0:
                video_path.unlink(missing_ok=True)
                if interrupted:
                    break
                raise OSError("No frames were recorded; the empty clip was discarded")

            append_manifest(manifest_path, video_path, label, signer)
            latest = (video_path, label)
            print(f"Saved {video_path}. Review it before the next clip.")

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                cv2.putText(
                    frame,
                    "r=review last  d=delete last  any other key=continue",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 0),
                    2,
                )
                cv2.imshow("ISL clip recorder", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r") and latest:
                    play_clip(latest[0], cv2)
                elif key == ord("d") and latest:
                    latest[0].unlink(missing_ok=True)
                    remove_manifest_entry(manifest_path, latest[0])
                    print(f"Deleted {latest[0]} and removed its manifest row.")
                    latest = None
                    break
                elif key != 255:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signer-id", required=True, help="Pseudonymous signer ID")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/own"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/clips.csv"))
    args = parser.parse_args()
    record(
        args.signer_id,
        args.output_root,
        args.manifest,
        args.camera_index,
        args.duration,
        args.countdown,
    )


if __name__ == "__main__":
    main()
