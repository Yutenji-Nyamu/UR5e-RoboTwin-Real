#!/usr/bin/env python3
"""Archive small experiment evidence for Git without altering live data or training."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


MAX_FILE_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".md", ".log"}
SMALL_ARRAY_SUFFIXES = {".npz", ".npy"}
METADATA_NAMES = {"_CHECKPOINT_METADATA", "_METADATA", "_sharding", ".zattrs", ".zarray", ".zgroup", ".zmetadata"}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def candidates(root: Path):
    """Do not follow directory/file symlinks into unrelated trees."""
    if root.is_symlink():
        return
    if root.is_file():
        yield root
    elif root.is_dir():
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                if not path.is_symlink():
                    yield path


def selected(path: Path, scope: str) -> bool:
    if path.name.startswith(".env") or path.suffix in {".pem", ".key"}:
        return False
    if scope == "all_small":
        return True
    return (
        path.suffix.lower() in TEXT_SUFFIXES | SMALL_ARRAY_SUFFIXES
        or path.name in METADATA_NAMES
        or (scope == "checkpoint" and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    )


def complete_jsonl_prefix(data: bytes) -> bytes:
    """Keep complete records if a live writer was sampled in its final JSON record."""
    if data and not data.endswith(b"\n"):
        tail_start = data.rfind(b"\n") + 1
        try:
            json.loads(data[tail_start:])
        except (ValueError, UnicodeDecodeError):
            return data[:tail_start]
    return data


def write_changed(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".archive-tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def archive(repository: Path, data_root: Path, output: Path, max_bytes: int = MAX_FILE_BYTES) -> dict:
    repository, data_root, output = repository.resolve(), data_root.resolve(), output.resolve()
    sources = [
        (repository / "logs", "repository/logs", "all_small"),
        (repository / "outputs", "repository/outputs", "all_small"),
        (repository / "checkpoints", "repository/checkpoints", "checkpoint"),
        (repository / "configs/lab.yaml", "repository/configs/lab.yaml", "all_small"),
        (data_root / "DATA_LOG.md", "data_root/DATA_LOG.md", "all_small"),
        (data_root / "raw", "data_root/raw", "metadata"),
        (data_root / "converted", "data_root/converted", "metadata"),
        (data_root / "pi05", "data_root/pi05", "metadata"),
        (data_root / "checkpoints", "data_root/checkpoints", "checkpoint"),
        (data_root / "logs", "data_root/logs", "all_small"),
    ]
    for source, _, _ in sources:
        if output == source or output.is_relative_to(source):
            raise ValueError("archive output must not be inside an evidence source")
    manifest_path = output / "MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {row["archive"]: row for row in previous.get("files", [])}
    observed, excluded = set(), Counter()
    captured = datetime.now(timezone.utc).isoformat()
    for source, prefix, scope in sources:
        for path in candidates(source):
            if not selected(path, scope):
                excluded["bulk_payload_or_non_evidence"] += 1
                continue
            stat = path.stat()
            if stat.st_size > max_bytes:
                excluded["over_size_limit"] += 1
                continue
            with path.open("rb") as handle:
                raw = handle.read(max_bytes + 1)
            if len(raw) > max_bytes:
                excluded["over_size_limit"] += 1
                continue
            payload = complete_jsonl_prefix(raw) if path.suffix == ".jsonl" else raw
            if path.suffix == ".json":
                try:
                    json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    excluded["incomplete_json_retry_next_snapshot"] += 1
                    continue
            relative = Path(prefix) if source.is_file() else Path(prefix) / path.relative_to(source)
            archive_name = relative.as_posix()
            write_changed(output / relative, payload)
            record = {
                "archive": archive_name,
                "source": str(path),
                "source_mtime_ns": stat.st_mtime_ns,
                "source_bytes_read": len(raw),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "archived_bytes": len(payload),
                "archived_sha256": hashlib.sha256(payload).hexdigest(),
                "incomplete_tail_bytes_omitted": len(raw) - len(payload),
            }
            old = records.get(archive_name, {})
            record["captured_at_utc"] = (
                old["captured_at_utc"]
                if all(old.get(key) == value for key, value in record.items())
                else captured
            )
            records[archive_name] = record
            observed.add(archive_name)
    manifest = {
        "schema_version": 1,
        "snapshot_started_at_utc": captured,
        "repository": str(repository),
        "data_root": str(data_root),
        "max_file_bytes": max_bytes,
        "scope": "per-file evidence snapshots; live runs may be incomplete; not a model/data backup",
        "sources": [{"path": str(path), "archive_prefix": prefix, "scope": scope} for path, prefix, scope in sources],
        "files": sorted(records.values(), key=lambda row: row["archive"]),
        "retained_from_earlier_snapshots": sorted(set(records) - observed),
        "excluded_counts": dict(sorted(excluded.items())),
        "file_count": len(records),
        "total_archived_bytes": sum(row["archived_bytes"] for row in records.values()),
    }
    write_changed(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=Path("/data/robotics/ur5e-real"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.repository / "docs/experiments/evidence"
    result = archive(args.repository, args.data_root, output)
    print(json.dumps({"output": str(output), "files": result["file_count"],
                      "MiB": round(result["total_archived_bytes"] / 1024**2, 2),
                      "excluded": result["excluded_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
