"""Atomic paired checkpoints. RNG and optimizer states are resume state, not weights only."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import torch

from .run import load_run, update_state, verify_artifact
from .storage import atomic_json, file_digest, read_json


def rng_state(generator):
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": generator.bit_generator.state,
    }


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.bit_generator.state = state["numpy"]


def save(directory, kind, step, payload, *, cache_id, token_sha256=None):
    directory, run, _ = load_run(directory)
    path = directory / "checkpoints" / kind / f"update_{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A resume from an older accepted version creates a new branch of history, never overwrites it.
    if path.exists():
        path = path.with_name(f"update_{step:08d}_{uuid.uuid4().hex[:8]}.pt")
    temporary = path.with_suffix(".tmp")
    metadata = {
        "version": 1,
        "kind": kind,
        "run_id": run["run_id"],
        "cache_id": cache_id,
        "token_sha256": token_sha256,
        "step": step,
    }
    if kind == "heads":
        metadata["counters"] = {
            key: payload["learner"][key] for key in ("bc_updates", "critic_updates", "actor_updates")
        }
        metadata["online_start_update"] = payload["learner"]["online_start_update"]
    try:
        with temporary.open("xb") as handle:
            torch.save({"metadata": metadata, **payload}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        atomic_json(str(path) + ".json", {**metadata, "file_sha256": file_digest(path)}, replace=False)
    finally:
        temporary.unlink(missing_ok=True)
    reference = {"path": str(path), "sha256": file_digest(path)}
    update_state(directory, **{"latest_token" if kind == "token" else "latest_head": reference})
    return path


def load(reference, run, cache, *, kind, token_sha256=None, device="cpu"):
    path = verify_artifact(reference)
    metadata = read_json(str(path) + ".json")
    expected = {
        "version": 1,
        "kind": kind,
        "run_id": run["run_id"],
        "cache_id": cache["cache_id"],
        "token_sha256": token_sha256,
    }
    if any(metadata.get(k) != v for k, v in expected.items()) or metadata["file_sha256"] != reference["sha256"]:
        raise ValueError("checkpoint lineage/normalization/token identity mismatch")
    value = torch.load(path, map_location=device, weights_only=True)
    if value["metadata"] != {k: v for k, v in metadata.items() if k != "file_sha256"}:
        raise ValueError("checkpoint metadata mismatch")
    return value


def prune_heads(directory):
    """Only remove generated, unselected old head snapshots; keep all round/decision references."""
    directory, run, state = load_run(directory)
    keep = {v["path"] for k, v in state.items() if k in ("selected_head", "latest_head") and v}
    for path in (directory / "rounds").glob("*/spec.json"):
        ref = read_json(path).get("heads")
        if ref:
            keep.add(ref["path"])
    import json

    decisions = directory / "decisions.jsonl"
    if decisions.exists():
        for line in decisions.read_text().splitlines():
            ref = json.loads(line).get("checkpoint")
            if ref:
                keep.add(ref["path"])
    files = sorted((directory / "checkpoints" / "heads").glob("update_*.pt"), key=lambda p: p.stat().st_mtime_ns)
    for path in files[: -run["config"]["save"]["keep_last"]]:
        if str(path) not in keep:
            path.unlink()
            Path(str(path) + ".json").unlink(missing_ok=True)
