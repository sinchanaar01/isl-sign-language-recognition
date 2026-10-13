import csv
import tempfile
import unittest
from pathlib import Path

from tools.make_manifest import make_manifest
from tools.record_clips import (
    append_manifest,
    next_clip_number,
    remove_manifest_entry,
    safe_component,
)


class ClipToolTests(unittest.TestCase):
    def test_safe_component_rejects_empty_and_normalizes_path_separators(self):
        self.assertEqual(safe_component("HELLO PLEASE", "label"), "HELLO_PLEASE")
        with self.assertRaises(ValueError):
            safe_component("../", "label")

    def test_manifest_append_and_targeted_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            first.touch()
            second.touch()
            manifest = root / "clips.csv"
            append_manifest(manifest, first, "HELLO", "s1")
            append_manifest(manifest, second, "WATER", "s1")
            remove_manifest_entry(manifest, first)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "WATER")

    def test_next_clip_number_ignores_other_signs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "HELLO_001.mp4").touch()
            (root / "WATER_012.mp4").touch()
            self.assertEqual(next_clip_number(root, "HELLO"), 2)

    def test_make_manifest_validates_and_keeps_signers_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "clips.csv"
            rows = []
            for signer, labels in (
                ("s1", ("HELLO", "WATER")),
                ("s2", ("HELLO", "WATER")),
                ("s3", ("HELLO", "WATER")),
            ):
                for index, label in enumerate(labels):
                    video = root / f"{signer}_{index}.mp4"
                    video.touch()
                    rows.append(
                        {
                            "video_path": str(video),
                            "label": label,
                            "signer_id": signer,
                        }
                    )
            with clips.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("video_path", "label", "signer_id")
                )
                writer.writeheader()
                writer.writerows(rows)
            manifest = root / "processed.csv"
            report = root / "summary.json"
            summary = make_manifest(
                clips, manifest, report, video_probe=lambda path: True
            )
            self.assertEqual(summary["clip_count"], 6)
            self.assertEqual(summary["signer_count"], 3)
            with manifest.open(newline="", encoding="utf-8") as handle:
                result = list(csv.DictReader(handle))
            assignments = {}
            for row in result:
                assignments.setdefault(row["signer_id"], set()).add(row["split"])
            self.assertTrue(all(len(values) == 1 for values in assignments.values()))
            self.assertEqual({row["split"] for row in result}, {"train", "val", "test"})

    def test_make_manifest_fails_for_missing_signer_or_corrupt_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "clips.csv"
            videos = []
            rows = []
            for index, signer in enumerate(("s1", "s2", "s3")):
                video = root / f"{index}.mp4"
                video.touch()
                videos.append(video)
                rows.append(
                    {
                        "video_path": str(video),
                        "label": "HELLO",
                        "signer_id": signer,
                    }
                )
            with clips.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("video_path", "label", "signer_id")
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "corrupt/unreadable"):
                make_manifest(
                    clips,
                    root / "processed.csv",
                    root / "summary.json",
                    video_probe=lambda path: path != videos[0],
                )
            rows[-1]["signer_id"] = "s2"
            with clips.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("video_path", "label", "signer_id")
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "at least 3"):
                make_manifest(
                    clips,
                    root / "processed.csv",
                    root / "summary.json",
                    video_probe=lambda path: True,
                )


if __name__ == "__main__":
    unittest.main()
