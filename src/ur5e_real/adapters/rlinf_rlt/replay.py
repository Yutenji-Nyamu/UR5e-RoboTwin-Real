"""Durable episode journal and human-labelled macro-action replay."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .run import load_run
from .storage import append_json, atomic_json, atomic_npz, digest, file_digest, read_json

LABELS = ("success", "failure", "timeout", "aborted")


def validate_transition(data, config):
    k, z = config["real"]["action_steps"], config["token"]["embed_dim"]
    shapes = {
        "z": (z,),
        "next_z": (z,),
        "proprio": (13,),
        "next_proprio": (13,),
        "reference": (k, 7),
        "next_reference": (k, 7),
        "action": (k, 7),
    }
    for key, shape in shapes.items():
        value = np.asarray(data[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid replay {key}: expected {shape} finite values")
    if any(np.max(np.abs(data[key])) > 1.0 for key in ("reference", "next_reference", "action")):
        raise ValueError("replay actions must use the frozen canonical coordinate system")
    if np.asarray(data["lengths"]).shape != () or not 1 <= int(data["lengths"]) <= k:
        raise ValueError("replay needs the actual nonzero executed prefix length")


class EpisodeWriter:
    def __init__(self, path, spec, config):
        self.path, self.config = Path(path), config
        self.path.mkdir(parents=True, exist_ok=False)
        self.spec = spec
        self.entries = []
        atomic_json(
            self.path / "episode.json",
            {"run_id": spec["run_id"], "round_id": spec["round_id"], "spec_sha256": digest(spec), "status": "pending"},
            replace=False,
        )

    def event(self, kind, **values):
        append_json(self.path / "events.jsonl", {"event": kind, **values})

    def transition(self, current, following, executed_k, *, observation, next_observation, trace):
        data = {k: np.asarray(current[k], dtype=np.float32) for k in ("z", "proprio", "reference", "action")}
        data.update({f"next_{k}": np.asarray(following[k], dtype=np.float32) for k in ("z", "proprio", "reference")})
        data["lengths"] = np.asarray(executed_k, dtype=np.int64)
        validate_transition(data, self.config)
        path = self.path / f"transition_{len(self.entries):04d}.npz"
        raw = {}
        for prefix, obs in (("obs", observation), ("next_obs", next_observation)):
            raw[f"{prefix}_state"] = np.asarray(obs["state"])
            raw[f"{prefix}_executor_aux"] = np.asarray(obs["executor_aux"])
            for key in ("rlt_observed_at", "rlt_controller_time_s", "rlt_state_host_time_s"):
                if key in obs:
                    raw[f"{prefix}_{key}"] = np.asarray(obs[key])
            for key, value in obs["images"].items():
                raw[f"{prefix}_{key}"] = value
        atomic_npz(path, **data, **raw, **{f"trace_{k}": np.asarray(v) for k, v in trace.items()})
        self.entries.append({"file": path.name, "sha256": file_digest(path)})
        atomic_json(self.path / "index.json", self.entries)
        self.event("transition", index=len(self.entries) - 1, executed_k=executed_k)

    def finish(self, label, *, reason, elapsed_s):
        return label_episode(self.path, label, reason=reason, elapsed_s=elapsed_s)


def label_episode(path, label, *, reason, elapsed_s=None):
    path = Path(path)
    if label not in LABELS or not reason.strip():
        raise ValueError("choose success/failure/timeout/aborted with a result note")
    read_json(path / "episode.json")
    if label != "aborted":
        # Only a normally ended capture has a final pre-reset observation and can be labelled for TD.
        capture = read_json(path / "capture_complete.json")
        if not capture["transitions"] or (label == "timeout" and capture["stop_reason"] != "budget"):
            raise ValueError("no valid rollout, or timeout requested for a non-time-limit termination")
        if elapsed_s is None:
            elapsed_s = capture["elapsed_s"]
    result = {"label": label, "reason": reason, "elapsed_s": elapsed_s, "eligible_for_td": label != "aborted"}
    atomic_json(path / "result.json", result, replace=False)
    append_json(path / "events.jsonl", {"event": "human_result", **result})
    return result


def summarize(round_dir):
    round_dir = Path(round_dir)
    spec = read_json(round_dir / "spec.json")
    episodes = sorted(round_dir.glob("episode_*"))
    results = []
    for path in episodes:
        results.append(read_json(path / "result.json") if (path / "result.json").exists() else {"label": "pending"})
    summary = {
        "run_id": spec["run_id"],
        "round_id": spec["round_id"],
        "kind": spec["kind"],
        "spec_sha256": digest(spec),
        "attempts": len(episodes),
        **{label: sum(r["label"] == label for r in results) for label in (*LABELS, "pending")},
        "episodes": [{"path": str(p), **r} for p, r in zip(episodes, results, strict=True)],
    }
    summary["recorded_transitions"] = sum(
        len(read_json(p / "index.json")) for p in episodes if (p / "index.json").exists()
    )
    summary["eligible_transitions"] = sum(
        len(read_json(p / "index.json"))
        for p, r in zip(episodes, results, strict=True)
        if r["label"] in ("success", "failure", "timeout")
    )
    atomic_json(round_dir / "summary.json", summary)
    return summary


def load_replay(directory):
    directory, run, state = load_run(directory)
    batches, identities = [], []
    for spec_path in sorted((directory / "rounds").glob("*/spec.json")):
        spec = read_json(spec_path)
        if spec["run_id"] != run["run_id"] or spec["token"] != state["selected_token"]:
            raise ValueError("replay representation identity mismatch")
        for episode in sorted(spec_path.parent.glob("episode_*")):
            if not (episode / "result.json").exists():
                continue
            result = read_json(episode / "result.json")
            if result["label"] == "aborted":
                continue
            identity = read_json(episode / "episode.json")
            if identity["spec_sha256"] != digest(spec):
                raise ValueError("episode was captured with a different round specification")
            entries = read_json(episode / "index.json")
            complete = read_json(episode / "capture_complete.json")
            if complete["transitions"] != len(entries) or not entries:
                raise ValueError("episode has an incomplete transition journal")
            identities.append(
                {"episode": str(episode), "result_sha256": file_digest(episode / "result.json"), "transitions": entries}
            )
            for i, entry in enumerate(entries):
                path = episode / entry["file"]
                if file_digest(path) != entry["sha256"]:
                    raise ValueError("transition archive has changed")
                keys = ("z", "proprio", "reference", "action", "next_z", "next_proprio", "next_reference", "lengths")
                with np.load(path, allow_pickle=False) as data:
                    item = {key: data[key].copy() for key in keys}
                validate_transition(item, run["config"])
                final = i == len(entries) - 1
                rewards = np.zeros(run["config"]["real"]["action_steps"], dtype=np.float32)
                if final and result["label"] == "success":
                    rewards[int(item["lengths"]) - 1] = 1.0
                item.update(
                    rewards=rewards,
                    terminated=np.asarray(final and result["label"] != "timeout"),
                    truncated=np.asarray(final and result["label"] == "timeout"),
                )
                batches.append(item)
    if not batches:
        raise ValueError("no committed, non-aborted real transitions; offline demos are BC-only")
    return {key: np.stack([b[key] for b in batches]) for key in batches[0]}, {
        "sha256": digest(identities),
        "episodes": len(identities),
        "transitions": len(batches),
        "sources": identities,
    }
