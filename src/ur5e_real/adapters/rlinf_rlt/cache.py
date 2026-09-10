"""Offline prefix cache: no robot connections and no invented RL transitions."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..robotwin_pi05.contract import STATE_LAYOUT
from ..robotwin_pi05.dataset import offline_observation
from .actions import ActionCodec, proprio
from .config import FEATURE_VERSION
from .run import load_run, update_state, validate_base
from .storage import atomic_json, atomic_npz, read_json, digest, file_digest


def validate_features(features, token_config):
    if features.get("version") != FEATURE_VERSION:
        raise ValueError("frozen feature protocol mismatch")
    prefix = np.asarray(features["prefix"], dtype=np.float32)
    mask = np.asarray(features["mask"])
    if prefix.shape != (token_config["prefix_seq_len"], token_config["input_dim"]):
        raise ValueError(f"actual prefix shape {prefix.shape} differs from the configured token model")
    if mask.dtype != bool or mask.shape != prefix.shape[:1] or not mask.any():
        raise ValueError("prefix mask must retain real image tokens")
    for key in ("prefix", "raw_reference", "normalized_state"):
        if not np.isfinite(np.asarray(features[key])).all():
            raise ValueError(f"non-finite {key}")
    if np.asarray(features["raw_reference"]).shape != (50, 32):
        raise ValueError("raw native reference must retain H50 and padded dimension 32")
    if np.asarray(features["normalized_state"]).shape != (32,):
        raise ValueError("native normalized state must retain padded dimension 32")
    return prefix, mask


def build_cache(directory, *, limit=None, policy=None):
    directory, run, state = load_run(directory)
    if state["stage"] != "cache":
        raise ValueError("feature cache is immutable after completion")
    dataset = Path(run["dataset"])
    cache_dir = directory / "cache"
    cache_dir.mkdir(exist_ok=True)
    if (cache_dir / "ready.json").is_file():
        completed = load_cache(directory)
        update_state(directory, stage="token")
        return completed  # Recover a crash between manifest publication and state publication.
    index_path = cache_dir / "index.json"
    entries = read_json(index_path) if index_path.exists() else []
    if policy is None:
        validate_base(run)
        from ..robotwin_pi05.native import add_native_paths

        add_native_paths()
        from ..robotwin_pi05.serve import load_policy
        from ..robotwin_pi05.rlt_features import FrozenFeaturePolicy

        native, _, _ = load_policy(dataset, Path(run["checkpoint"]))
        policy = FrozenFeaturePolicy(native)
    locations = []
    for run_id in run["contract"]["run_ids"]:
        with np.load(dataset / "ur5e_adapter" / f"{run_id}.npz", allow_pickle=False) as archive:
            locations.extend((run_id, i) for i in range(len(archive["vectors"]) - 1))
    total_available = len(locations)
    if limit is not None:
        if limit < 1:
            raise ValueError("cache limit must be positive")
        locations = locations[:limit]
    if len(entries) > len(locations):
        raise ValueError("resume limit is smaller than the existing cache")
    for i, (run_id, frame) in enumerate(locations):
        if i < len(entries):
            if file_digest(cache_dir / entries[i]["file"]) != entries[i]["sha256"]:
                raise ValueError("cached features were modified")
            continue
        obs, _ = offline_observation(dataset, run_id, frame)
        noise = np.random.default_rng(run["config"]["token_train"]["seed"] + i).normal(size=(50, 32)).astype(np.float32)
        result = policy.infer(obs, noise=noise)
        features = result["rlt_features"]
        prefix, mask = validate_features(features, run["config"]["token"])
        path = cache_dir / f"frame_{i:06d}.npz"
        atomic_npz(
            path,
            prefix=prefix,
            mask=mask,
            raw_reference=features["raw_reference"],
            proprio=proprio(features["normalized_state"]),
            physical_reference=result["actions"],
            observation_q=np.asarray(obs["state"][:6], dtype=np.float64),
        )
        entries.append({"file": path.name, "sha256": file_digest(path), "run_id": run_id, "frame": frame})
        atomic_json(index_path, entries)
        print(f"[CACHE] {i + 1}/{len(locations)} {run_id}:{frame}", flush=True)
    references = []
    for entry in entries:
        with np.load(cache_dir / entry["file"], allow_pickle=False) as sample:
            references.append(sample["raw_reference"])
    norm = read_json(dataset / "ur5e_adapter" / "assets" / STATE_LAYOUT / "norm_stats.json")
    codec = ActionCodec.fit(np.stack(references), norm)
    parity_max = 0.0
    for entry in entries:
        with np.load(cache_dir / entry["file"], allow_pickle=False) as sample:
            decoded = codec.decode(codec.encode(sample["raw_reference"]), sample["observation_q"])
            expected = sample["physical_reference"][:, [0, 1, 2, 3, 4, 5, 13]].copy()
            expected[:, 6] = np.clip(expected[:, 6], 0, 1)
            parity_max = max(parity_max, float(np.abs(decoded - expected).max()))
    if not np.isfinite(parity_max) or parity_max > 1e-4:
        raise ValueError(f"canonical/native decode parity failed: {parity_max}")
    manifest = {
        "run_id": run["run_id"],
        "base_id": run["base_id"],
        "feature_version": FEATURE_VERSION,
        "files": entries,
        "codec": codec.as_dict(),
        "full_dataset": len(entries) == total_available,
        "source_observations": total_available,
        "decode_parity_max": parity_max,
        "offline_executor_aux": "unobserved zeros; BC only, never use as real TD replay",
        "reference_noise": "fixed Gaussian seed=token_train.seed+observation_index; reproducible cache resume",
    }
    manifest["cache_id"] = digest(manifest)
    atomic_json(cache_dir / "ready.json", manifest, replace=False)
    update_state(directory, stage="token")
    return manifest


def load_cache(directory, *, verify_files=True):
    directory, run, _ = load_run(directory)
    cache = read_json(directory / "cache" / "ready.json")
    if (
        cache["run_id"] != run["run_id"]
        or digest({k: v for k, v in cache.items() if k != "cache_id"}) != cache["cache_id"]
    ):
        raise ValueError("cache identity differs from this run")
    if verify_files:
        for entry in cache["files"]:
            if file_digest(directory / "cache" / entry["file"]) != entry["sha256"]:
                raise ValueError("cached observation was modified")
    return cache
