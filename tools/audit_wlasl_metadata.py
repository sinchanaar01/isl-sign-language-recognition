"""Audit downloaded WLASL annotations without downloading source videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/"
    "ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/WLASL_v0.3.json"
)


def audit_metadata(
    metadata_path: Path,
    output_path: Path = Path("reports/runs/wlasl_metadata_audit.json"),
) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    raw = metadata_path.read_bytes()
    categories = json.loads(raw)
    if not isinstance(categories, list):
        raise ValueError("Expected the WLASL metadata root to be a JSON list")
    instance_counts = Counter()
    signer_ids: set[str] = set()
    video_ids: set[str] = set()
    missing_fields = Counter()
    for category in categories:
        if not isinstance(category, dict) or "gloss" not in category:
            raise ValueError("A WLASL category is missing its gloss field")
        for instance in category.get("instances", []):
            instance_counts[str(category["gloss"])] += 1
            if instance.get("signer_id") is not None:
                signer_ids.add(str(instance["signer_id"]))
            if instance.get("video_id") is not None:
                video_ids.add(str(instance["video_id"]))
            for field in ("signer_id", "video_id", "url", "split"):
                if instance.get(field) is None:
                    missing_fields[field] += 1
    stat = metadata_path.stat()
    result = {
        "audit_kind": "metadata_only_not_training_result",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "local_path": str(metadata_path.resolve()),
        "file_size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "gloss_category_count": len(categories),
        "annotation_instance_count": sum(instance_counts.values()),
        "unique_signer_ids_in_metadata": len(signer_ids),
        "unique_video_ids_in_metadata": len(video_ids),
        "instances_missing_fields": dict(missing_fields),
        "video_media_downloaded": False,
        "training_samples": 0,
        "language": "American Sign Language (ASL)",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/WLASL_v0.3.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/runs/wlasl_metadata_audit.json")
    )
    args = parser.parse_args()
    audit_metadata(args.metadata, args.output)


if __name__ == "__main__":
    main()
