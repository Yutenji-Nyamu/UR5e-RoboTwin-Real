"""Verify the tracked, patched source snapshots without a nested Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
LOCK = REPOSITORY / "integrations/vendor_sources.lock.json"


def verify_source(name: str, *, root: Path | None = None, prefix: str = "", lock: Path = LOCK) -> dict:
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise ValueError("unsupported vendor source lock")
    item = manifest["sources"][name]
    root = (root or REPOSITORY / item["directory"]).resolve()
    expected = {k: v for k, v in item["files"].items() if not prefix or k.startswith(prefix.rstrip("/") + "/")}
    if not expected:
        raise ValueError("vendor verification matched no files")
    changed = []
    for relative, info in expected.items():
        path = root / relative
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != info["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]):
            changed.append(relative)
    # Extra Python files can alter module resolution even if pinned files match.
    source_root = root / prefix
    for path in source_root.rglob("*.py"):
        if {"__pycache__", ".git", ".venv"}.intersection(path.parts):
            continue
        if path.relative_to(root).as_posix() not in expected:
            changed.append(path.relative_to(root).as_posix())
    if changed:
        raise RuntimeError(f"{name} source differs from vendor lock: {', '.join(changed[:10])}")
    return {"source": name, "files": len(expected), "bytes": sum(f["bytes"] for f in expected.values()),
            "upstream_commit": item.get("upstream_commit"), "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("RoboTwin", "MetisWAM4D", "all"), default="all", nargs="?")
    args = parser.parse_args()
    names = ("RoboTwin", "MetisWAM4D") if args.source == "all" else (args.source,)
    print(json.dumps([verify_source(name) for name in names], indent=2))


if __name__ == "__main__":
    main()
