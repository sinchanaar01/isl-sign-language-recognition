"""Attempt licensed INCLUDE/Kaggle acquisition without guessing rights or signers.

Downloaded media is kept under data/raw/downloaded and never merged with the
consented live-demo recordings in data/raw/own.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from tools.make_manifest import video_is_readable

INCLUDE_OFFICIAL_PAGE = "https://www.iitm.ac.in/donlab/INCLUDE"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_KAGGLE_LICENSES = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "odc-by-1.0",
    "mit",
    "apache-2.0",
}
ALLOWED_INCLUDE_LICENSES = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "odc-by-1.0",
    "mit license",
    "apache-2.0",
}
CSV_FIELDS = ("video_path", "label", "signer_id")


def normalize_class(value: str) -> str:
    """Normalize dataset folder and target names for exact class matching."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def read_targets(path: Path) -> dict[str, str]:
    """Return normalized candidate class names mapped to their display labels."""
    targets: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        label = line.strip()
        if label and not label.startswith("#"):
            targets[normalize_class(label)] = label
    return targets


def free_disk_bytes(path: Path) -> int:
    """Return available bytes on the filesystem containing path."""
    return shutil.disk_usage(path).free


def require_space(destination: Path, dataset_bytes: int) -> None:
    """Fail before network transfer when free space cannot hold the source."""
    if dataset_bytes <= 0:
        raise ValueError("Dataset size must be known and positive before download.")
    destination.mkdir(parents=True, exist_ok=True)
    free = free_disk_bytes(destination)
    if free < dataset_bytes:
        raise OSError(
            f"Insufficient free disk space: {free} bytes free, "
            f"{dataset_bytes} bytes required before extraction."
        )


def fetch_url(url: str, timeout: int = 20) -> tuple[bytes, str]:
    """Fetch a public page or file and return its bytes and final URL."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "ISL-recognition-data-audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.geturl()


def extract_download_links(page: bytes, page_url: str) -> list[tuple[str, str]]:
    """Find media/archive links and an adjacent human-readable link label."""
    text = page.decode("utf-8", errors="replace")
    matches = re.findall(
        r"""<a\b[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    links: list[tuple[str, str]] = []
    for href, label in matches:
        label_text = re.sub(r"<[^>]+>", " ", label)
        label_text = re.sub(r"\s+", " ", label_text).strip()
        full_url = urllib.parse.urljoin(page_url, href)
        if Path(urllib.parse.urlparse(full_url).path).suffix.casefold() in {
            ".zip",
            ".tar",
            ".gz",
            ".7z",
        }:
            links.append((full_url, label_text))
    return links


def explicit_license(page: bytes) -> str | None:
    """Accept only a clearly named license string from the official page."""
    text = re.sub(r"<[^>]+>", " ", page.decode("utf-8", errors="replace"))
    text = re.sub(r"\s+", " ", text)
    match = re.search(
        r"\b(CC0(?:\s*1\.0)?|CC[- ]BY(?:[- ]SA)?(?:[- ]NC)?(?:[- ]ND)?"
        r"(?:\s*4\.0)?|Apache[- ]2\.0|MIT License|ODC[- ]BY[- ]1\.0)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else None


def fetch_content_length(url: str) -> int | None:
    """Read a remote file size without downloading its contents."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ISL-recognition-data-audit/1.0"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            value = response.headers.get("Content-Length")
    except (urllib.error.URLError, TimeoutError):
        return None
    return int(value) if value and value.isdigit() else None


def safe_extract_matching(
    archive_path: Path,
    destination: Path,
    target_names: set[str],
) -> int:
    """Extract only top-level category folders whose names match verified targets."""
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or not path.parts:
                continue
            matching_class = next(
                (
                    part
                    for part in path.parts[:-1]
                    if normalize_class(part) in target_names
                ),
                None,
            )
            if matching_class is None:
                continue
            output = (destination / Path(*path.parts)).resolve()
            if destination.resolve() not in output.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    return extracted


def _metadata_signers(root: Path) -> dict[Path, tuple[str, str]]:
    """Read only canonical video_path,label,signer_id metadata if present."""
    result: dict[Path, tuple[str, str]] = {}
    for metadata_path in root.rglob("*.csv"):
        if metadata_path.name.casefold() == "clips.csv":
            continue
        try:
            with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not set(CSV_FIELDS).issubset(
                    reader.fieldnames
                ):
                    continue
                for row in reader:
                    video = (row.get("video_path") or "").strip()
                    label = (row.get("label") or "").strip()
                    signer = (row.get("signer_id") or "").strip()
                    if video and label and signer:
                        source_video = Path(video)
                        if not source_video.is_absolute():
                            source_video = metadata_path.parent / source_video
                        result[source_video.resolve()] = (label, signer)
        except (OSError, csv.Error, UnicodeError):
            continue
    return result


def build_clips_manifest(
    dataset_root: Path,
    output_csv: Path,
    target_names: dict[str, str],
    readable_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Index target-class videos and report missing signer IDs without guessing."""
    signer_metadata = _metadata_signers(dataset_root)
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    missing_signer_count = 0
    for video in sorted(dataset_root.rglob("*")):
        if not video.is_file() or video.suffix.casefold() not in VIDEO_SUFFIXES:
            continue
        if readable_paths is not None and str(video) not in readable_paths:
            continue
        relative = video.relative_to(dataset_root)
        class_label = next(
            (
                target_names[normalize_class(part)]
                for part in relative.parts[:-1]
                if normalize_class(part) in target_names
            ),
            None,
        )
        metadata = signer_metadata.get(video.resolve())
        if metadata and normalize_class(metadata[0]) in target_names:
            class_label = target_names[normalize_class(metadata[0])]
            signer_id = metadata[1]
        else:
            signer_id = ""
        if not class_label:
            continue
        if not signer_id:
            missing_signer_count += 1
        counts[class_label] = counts.get(class_label, 0) + 1
        rows.append(
            {
                "video_path": str(video.resolve()),
                "label": class_label,
                "signer_id": signer_id,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "video_count": len(rows),
        "class_counts": counts,
        "signer_metadata_missing_count": missing_signer_count,
        "manifest_path": str(output_csv),
    }


def check_videos(root: Path) -> dict[str, Any]:
    """Decode a frame from every video; fail clearly if OpenCV is unavailable."""
    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )
    readable: list[str] = []
    corrupt: list[str] = []
    for video in videos:
        (readable if video_is_readable(video) else corrupt).append(str(video))
    return {
        "video_count": len(videos),
        "readable_count": len(readable),
        "corrupt_count": len(corrupt),
        "readable_files": readable,
        "corrupt_files": corrupt,
    }


def try_include(target_names: dict[str, str], destination: Path) -> dict[str, Any]:
    """Attempt the cited IIT Madras INCLUDE project page, failing closed."""
    result: dict[str, Any] = {
        "source": "INCLUDE",
        "status": "blocked",
        "source_page": INCLUDE_OFFICIAL_PAGE,
    }
    page, final_url = fetch_url(INCLUDE_OFFICIAL_PAGE)
    result["source_page"] = final_url
    license_name = explicit_license(page)
    if not license_name or normalize_class(license_name) not in {
        normalize_class(name) for name in ALLOWED_INCLUDE_LICENSES
    }:
        result["reason"] = (
            "Official page did not expose a recognized redistribution-compatible "
            "license; manual license/access review is required."
        )
        return result
    result["license"] = license_name
    archives = extract_download_links(page, final_url)
    if not archives:
        result["reason"] = (
            "Official page has no direct public dataset archive; sign-in or "
            "license click-through may be required."
        )
        return result
    matched_archives = [
        (url, label)
        for url, label in archives
        if any(
            normalize_class(target) in normalize_class(label)
            for target in target_names.values()
        )
    ]
    if not matched_archives:
        result["reason"] = (
            "No class-specific downloadable archives matched target_glosses.txt; "
            "refusing to download a full unfiltered archive."
        )
        return result
    for url, label in matched_archives:
        size = fetch_content_length(url)
        if size is None:
            result["reason"] = (
                f"Could not determine archive size before download: {url}"
            )
            return result
        require_space(destination, size)
        archive_path = destination / Path(urllib.parse.urlparse(url).path).name
        content, _ = fetch_url(url, timeout=120)
        if len(content) != size:
            result["reason"] = (
                f"Downloaded archive size differs from advertised Content-Length: {url}"
            )
            return result
        archive_path.write_bytes(content)
        extracted = safe_extract_matching(
            archive_path, destination / "include", set(target_names)
        )
        archive_path.unlink()
        if extracted:
            result.update(
                status="downloaded",
                dataset_url=url,
                archive_label=label,
                archive_size_bytes=size,
                extracted_video_files=extracted,
            )
            return result
    result["reason"] = "No matching target-class media found in eligible archives."
    return result


def try_kaggle(
    target_names: dict[str, str], destination: Path, dataset_slug: str | None
) -> dict[str, Any]:
    """Use Kaggle CLI only when credentials and an explicitly licensed slug exist."""
    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    if not credential_path.is_file():
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": (
                "Missing ~/.kaggle/kaggle.json (the home-relative credential path). "
                "Configure Kaggle API credentials "
                "following https://github.com/Kaggle/kaggle-api#api-credentials."
            ),
        }
    executable = shutil.which("kaggle")
    if not executable:
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": (
                "Kaggle credentials exist but CLI is missing. Run "
                "`python -m pip install kaggle`, then retry."
            ),
        }
    if not dataset_slug:
        discovery = subprocess.run(
            [
                executable,
                "datasets",
                "list",
                "-s",
                "Indian Sign Language",
                "--csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if discovery.returncode:
            return {
                "source": "Kaggle",
                "status": "blocked",
                "reason": discovery.stderr.strip() or discovery.stdout.strip(),
            }
        candidates = list(csv.DictReader(discovery.stdout.splitlines()))
        reviewed = [
            row
            for row in candidates
            if "indian sign language" in (row.get("title") or "").casefold()
            and (row.get("licenseName") or "").casefold() in ALLOWED_KAGGLE_LICENSES
        ]
        if not reviewed:
            return {
                "source": "Kaggle",
                "status": "blocked",
                "reason": (
                    "Kaggle search found no dataset with an exact ISL title and "
                    "recognized redistribution-compatible license. Review results "
                    "manually; no dataset was downloaded."
                ),
                "search_result_count": len(candidates),
                "dataset_titles": [row.get("title") for row in candidates[:10]],
            }
        dataset_slug = reviewed[0].get("ref") or reviewed[0].get("datasetRef")
        if not dataset_slug:
            return {
                "source": "Kaggle",
                "status": "blocked",
                "reason": "Kaggle search result did not expose a dataset slug.",
            }

    metadata_dir = destination / "kaggle_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_result = subprocess.run(
        [executable, "datasets", "metadata", dataset_slug, "-p", str(metadata_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if metadata_result.returncode:
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": metadata_result.stderr.strip() or metadata_result.stdout.strip(),
        }
    metadata_files = list(metadata_dir.glob("*.json"))
    if not metadata_files:
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": "Kaggle metadata command returned no dataset metadata JSON.",
        }
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    raw_licenses = metadata.get("licenses", [])
    nested_license = (
        raw_licenses[0].get("name", "")
        if raw_licenses and isinstance(raw_licenses[0], dict)
        else ""
    )
    license_name = str(metadata.get("licenseName") or nested_license).strip()
    if license_name.casefold() not in ALLOWED_KAGGLE_LICENSES:
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": (
                f"Kaggle license {license_name or '(missing)'} is not in the "
                "explicitly reviewed allowlist. Review the dataset card and rights "
                "manually; no media was downloaded."
            ),
            "license": license_name or None,
        }
    listed = subprocess.run(
        [executable, "datasets", "files", "-d", dataset_slug, "--csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode:
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": listed.stderr.strip() or listed.stdout.strip(),
        }
    file_rows = list(csv.DictReader(listed.stdout.splitlines()))
    candidates: list[str] = []
    for item in file_rows:
        filename = item.get("name") or item.get("fileName") or ""
        if any(
            normalize_class(target) in normalize_class(PurePosixPath(filename).stem)
            for target in target_names.values()
        ):
            candidates.append(filename)
    if not candidates:
        return {
            "source": "Kaggle",
            "status": "blocked",
            "reason": (
                "No Kaggle file is class-specific by target_glosses.txt; refusing "
                "to download a full archive or guess internal archive membership."
            ),
            "license": license_name,
        }
    bytes_by_file = {
        item.get("name") or item.get("fileName"): int(item.get("size", 0) or 0)
        for item in file_rows
    }
    total_bytes = sum(bytes_by_file.get(name, 0) for name in candidates)
    require_space(destination, total_bytes)
    downloaded = 0
    for filename in candidates:
        download = subprocess.run(
            [
                executable,
                "datasets",
                "download",
                "-d",
                dataset_slug,
                "-f",
                filename,
                "-p",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if download.returncode:
            return {
                "source": "Kaggle",
                "status": "blocked",
                "reason": download.stderr.strip() or download.stdout.strip(),
                "license": license_name,
                "downloaded_files": downloaded,
            }
        archive_candidates = sorted(destination.glob(f"{Path(filename).name}*.zip"))
        for archive_path in archive_candidates:
            class_name = normalize_class(Path(filename).stem)
            if class_name in target_names:
                safe_extract_matching(
                    archive_path,
                    destination / "kaggle_selected" / Path(filename).stem,
                    {class_name},
                )
        downloaded += 1
    return {
        "source": "Kaggle",
        "status": "downloaded",
        "dataset_slug": dataset_slug,
        "dataset_url": f"https://www.kaggle.com/datasets/{dataset_slug}",
        "license": license_name,
        "class_files_requested": candidates,
        "advertised_size_bytes": total_bytes,
        "downloaded_files": downloaded,
    }


def run_acquisition(
    target_file: Path,
    destination: Path,
    output_csv: Path,
    kaggle_slug: str | None = None,
) -> dict[str, Any]:
    """Try INCLUDE first, then licensed Kaggle only if INCLUDE is blocked."""
    targets = read_targets(target_file)
    report: dict[str, Any] = {
        "kind": "dataset_acquisition_audit",
        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_glosses": list(targets.values()),
        "free_disk_bytes_before_attempt": free_disk_bytes(destination.parent),
        "sources": [],
    }
    try:
        include = try_include(targets, destination)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        include = {
            "source": "INCLUDE",
            "status": "blocked",
            "source_page": INCLUDE_OFFICIAL_PAGE,
            "reason": f"Official source request failed: {exc}",
        }
    report["sources"].append(include)
    if include.get("status") != "downloaded":
        report["sources"].append(try_kaggle(targets, destination, kaggle_slug))

    if any(source.get("status") == "downloaded" for source in report["sources"]):
        try:
            report["video_check"] = check_videos(destination)
            readable_paths = set(report["video_check"]["readable_files"])
            report["manifest"] = build_clips_manifest(
                destination, output_csv, targets, readable_paths
            )
            classes_found = sorted(report["manifest"]["class_counts"])
            if classes_found:
                target_file.write_text(
                    "# Classes found in the acquired dataset; signs still require "
                    "fluent-signer verification.\n" + "\n".join(classes_found) + "\n",
                    encoding="utf-8",
                )
                report["updated_target_glosses"] = classes_found
        except (ImportError, RuntimeError) as exc:
            report["post_download_blocker"] = (
                f"Media downloaded but the corrupt-video check could not run: {exc}. "
                "Install app dependencies and rerun `python -m tools.download_data`."
            )
            report["status"] = "BLOCKED - needs human"
        if report.get("manifest", {}).get("signer_metadata_missing_count", 0):
            report["signer_blocker"] = (
                "Signer IDs are missing for some or all downloaded samples. They "
                "were left blank, never inferred from filenames. Do not train until "
                "the provider supplies valid signer metadata."
            )
            report["status"] = "BLOCKED - needs human"
        elif report.get("manifest", {}).get("video_count", 0) < 1:
            report["status"] = "BLOCKED - needs human"
            report["manual_instructions"] = (
                "Media was downloaded but no supported, readable video clips were "
                "indexed. Inspect the source archive structure and label metadata "
                "before training."
            )
        else:
            report["status"] = (
                "downloaded; inspect signer/class coverage before training"
            )
    else:
        report["status"] = "BLOCKED - needs human"
        report["manual_instructions"] = (
            "Contact INCLUDE maintainers for an official media URL and license, or "
            "configure Kaggle API credentials, verify an exact dataset card/license "
            "and signer metadata, then rerun `python -m tools.download_data "
            "--kaggle-slug OWNER/SLUG`. See data/sources.md."
        )
    report["free_disk_bytes_after_attempt"] = free_disk_bytes(destination.parent)
    return report


def append_source_record(report: dict[str, Any], source_path: Path) -> None:
    """Record each actual acquisition attempt and any license/source evidence."""
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        source_path.write_text("# Dataset sources and licensing\n", encoding="utf-8")
    lines = [
        "",
        f"## Automated acquisition attempt: {report['attempted_at_utc']}",
        "",
        f"- Free disk before attempt: {report['free_disk_bytes_before_attempt']} bytes.",
    ]
    for source in report["sources"]:
        lines.append(f"- {source.get('source')}: **{source.get('status')}**.")
        if source.get("source_page"):
            lines.append(f"  Source page: {source['source_page']}")
        if source.get("dataset_url"):
            lines.append(f"  Dataset URL: {source['dataset_url']}")
        if source.get("dataset_slug"):
            lines.append(
                f"  Dataset URL: https://www.kaggle.com/datasets/"
                f"{source['dataset_slug']}"
            )
        if source.get("license"):
            lines.append(f"  License reported by source: {source['license']}.")
        if source.get("reason"):
            lines.append(f"  Result: {source['reason']}")
    if report.get("video_check"):
        check = report["video_check"]
        lines.append(
            f"- Media audit: {check['video_count']} files, "
            f"{check['readable_count']} readable, {check['corrupt_count']} corrupt."
        )
    if report.get("manifest"):
        manifest = report["manifest"]
        lines.append(
            f"- Indexed clips: {manifest['video_count']}; classes: "
            f"{json.dumps(manifest['class_counts'], ensure_ascii=False)}."
        )
        lines.append(
            f"- Missing signer metadata: {manifest['signer_metadata_missing_count']} "
            "(blank; not inferred)."
        )
    if report.get("manual_instructions"):
        lines.append(f"- Manual action: {report['manual_instructions']}")
    elif report.get("signer_blocker"):
        lines.append(f"- Manual action: {report['signer_blocker']}")
    with source_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("data/target_glosses.txt"))
    parser.add_argument("--destination", type=Path, default=Path("data/raw/downloaded"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/raw/downloaded/clips.csv")
    )
    parser.add_argument("--kaggle-slug")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_acquisition(
        args.targets, args.destination, args.manifest, args.kaggle_slug
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = (
        args.report or Path("reports/runs") / f"data_acquisition_{timestamp}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    append_source_record(report, Path("data/sources.md"))
    print(json.dumps(report, indent=2))
    print(f"Acquisition audit: {report_path}")
    if report.get("status") == "BLOCKED - needs human":
        print(report["manual_instructions"], file=sys.stderr)


if __name__ == "__main__":
    main()
