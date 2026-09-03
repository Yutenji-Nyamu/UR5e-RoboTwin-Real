from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


REVIEW_RESULTS = {"success", "failure", "aborted"}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the small mutable index; captured data files stay untouched."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("run_id"):
        raise ValueError(f"{path}: not a valid session manifest")
    return data


def review_session(path: Path, result: str, note: str | None = None) -> dict[str, Any]:
    if result not in REVIEW_RESULTS:
        raise ValueError(f"result must be one of {sorted(REVIEW_RESULTS)}")
    manifest = load_manifest(path)
    if manifest.get("recording_status") in {"initializing", "recording"}:
        raise RuntimeError(f"{path}: recording has not stopped")
    review = {
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "result": result,
        "note": note or None,
    }
    reviews = manifest.setdefault("reviews", [])
    if not isinstance(reviews, list):
        raise ValueError(f"{path}: reviews must be a list")
    reviews.append(review)
    manifest["outcome"] = result
    write_manifest(path, manifest)
    return review


def session_summaries(action_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    paths = sorted(action_dir.glob("session_*.json"), reverse=True)
    if limit > 0:
        paths = paths[:limit]
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            manifest = load_manifest(path)
        except (OSError, ValueError):
            continue
        counts = manifest.get("counts") or {}
        rows.append(
            {
                "run_id": str(manifest["run_id"]),
                "task": str(manifest.get("task") or "-"),
                "duration_s": manifest.get("duration_s"),
                "outcome": str(manifest.get("outcome") or "unreviewed"),
                "rtde_samples": counts.get("rtde_samples", "-"),
                "frame_pairs": counts.get("frame_pairs", "-"),
                "manifest": path,
            }
        )
    return rows


def print_session_summaries(action_dir: Path, limit: int = 20) -> None:
    rows = session_summaries(action_dir, limit)
    if not rows:
        print(f"No sessions found in {action_dir}")
        return
    print("RUN_ID                   TASK                 DURATION  RESULT       RTDE   FRAMES")
    for row in rows:
        duration = "-" if row["duration_s"] is None else f"{float(row['duration_s']):.1f}s"
        print(
            f"{row['run_id']:<24} "
            f"{row['task'][:20]:<20} "
            f"{duration:>8}  "
            f"{row['outcome']:<11} "
            f"{str(row['rtde_samples']):>6} "
            f"{str(row['frame_pairs']):>8}"
        )
