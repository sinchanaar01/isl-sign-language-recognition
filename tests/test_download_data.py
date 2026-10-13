"""Smoke tests for metadata/source decisions; no sign video is fabricated."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.download_data import (
    extract_download_links,
    normalize_class,
    read_targets,
    require_space,
    safe_extract_matching,
)


class DownloadDataTests(unittest.TestCase):
    def test_target_glosses_match_category_names_without_punctuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            target_file = Path(temporary) / "targets.txt"
            target_file.write_text("# Candidates\nTHANK YOU\nWATER\n", encoding="utf-8")
            targets = read_targets(target_file)

        self.assertEqual(targets[normalize_class("Thank_You")], "THANK YOU")
        self.assertEqual(targets[normalize_class("water")], "WATER")
        self.assertNotIn(normalize_class("GOOD MORNING"), targets)

    def test_download_link_parser_resolves_only_archive_links(self):
        page = (
            b'<a href="/data/HELLO.zip">HELLO clips</a>' b'<a href="/terms">Terms</a>'
        )
        self.assertEqual(
            extract_download_links(page, "https://example.org/project/"),
            [("https://example.org/data/HELLO.zip", "HELLO clips")],
        )

    def test_archive_extraction_keeps_only_target_categories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("HELLO/metadata.txt", "non-video smoke fixture")
                handle.writestr("WATER/metadata.txt", "non-video smoke fixture")
            output = root / "selected"
            count = safe_extract_matching(archive, output, {"hello"})

            self.assertEqual(count, 1)
            self.assertTrue((output / "HELLO" / "metadata.txt").is_file())
            self.assertFalse((output / "WATER").exists())

    def test_unknown_dataset_size_prevents_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            with self.assertRaisesRegex(ValueError, "size must be known"):
                require_space(destination, 0)

    def test_insufficient_disk_space_prevents_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "download"
            with patch("tools.download_data.free_disk_bytes", return_value=99):
                with self.assertRaisesRegex(OSError, "Insufficient free disk space"):
                    require_space(destination, 100)


if __name__ == "__main__":
    unittest.main()
