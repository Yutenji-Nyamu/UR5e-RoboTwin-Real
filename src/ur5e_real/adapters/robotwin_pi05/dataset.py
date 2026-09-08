"""Immutable local LeRobot export; labels are measured next-state joint targets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .contract import STATE_LAYOUT, read_contract, contract_digest
from .native import add_native_paths


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def rgb_image(path: Path):
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def future_actions(vectors, horizon=50):
    # Matches LeRobot delta_timestamps: action[t] is q[t+1], end padding repeats it.
    indices = np.minimum(np.arange(len(vectors) - 1)[:, None] + 1 + np.arange(horizon), len(vectors) - 1)
    return vectors[indices].copy()


def export_dataset(episodes, report: dict, lerobot_home: Path):
    lerobot_home = lerobot_home.resolve()
    root = lerobot_home / report["contract"]["dataset_id"]
    if root.exists():
        raise FileExistsError(f"refusing to replace an existing dataset: {root}; select a new versioned repo_id")
    os.environ["HF_LEROBOT_HOME"] = str(lerobot_home)
    add_native_paths()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.shared.normalize import RunningStats, save
    from openpi.transforms import DeltaActions, make_bool_mask

    first = rgb_image(episodes[0].head[0])
    height, width, _ = first.shape
    names = [*[f"q_{i}" for i in range(6)], "dummy_gripper", *[f"q_copy_{i}" for i in range(6)], "gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [names]},
        "action": {"dtype": "float32", "shape": (14,), "names": [names]},
        **{
            f"observation.images.{cam}": {
                "dtype": "image",
                "shape": (3, height, width),
                "names": ["channel", "height", "width"],
            }
            for cam in ("cam_high", "cam_right_wrist")
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=report["contract"]["dataset_id"],
        root=root,
        fps=10,
        robot_type="ur5e_joint",
        features=features,
        use_videos=False,
        image_writer_threads=2,
    )
    adapter = root / "ur5e_adapter"
    state_stats, action_stats = RunningStats(), RunningStats()
    delta = DeltaActions(make_bool_mask(6, -1, 6, -1))
    try:
        for episode in episodes:
            vectors = episode.vectors
            labels = future_actions(vectors)
            for i in range(len(vectors) - 1):
                images = {
                    cam: rgb_image(paths[i])
                    for cam, paths in (("cam_high", episode.head), ("cam_right_wrist", episode.wrist))
                }
                if any(im.shape != (height, width, 3) for im in images.values()):
                    raise ValueError("all head/wrist images must share the declared geometry")
                dataset.add_frame(
                    {
                        "observation.state": vectors[i],
                        "action": vectors[i + 1],
                        "task": report["contract"]["prompt"],
                        **{f"observation.images.{cam}": im for cam, im in images.items()},
                    }
                )
                transformed = delta({"state": vectors[i].copy(), "actions": labels[i].copy()})
                state_stats.update(transformed["state"][None])
                action_stats.update(transformed["actions"])
            dataset.save_episode()
            adapter.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                adapter / f"{episode.run_id}.npz",
                vectors=vectors,
                times=episode.times,
                head=np.asarray([str(p) for p in episode.head]),
                wrist=np.asarray([str(p) for p in episode.wrist]),
            )
        stats_dir = adapter / "assets" / STATE_LAYOUT
        save(stats_dir, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
        write_json(adapter / "contract.json", report["contract"])
        write_json(adapter / "audit.json", report)
        write_json(
            adapter / "ready.json",
            {
                "contract_sha256": contract_digest(report["contract"]),
                "norm_sha256": hashlib.sha256((stats_dir / "norm_stats.json").read_bytes()).hexdigest(),
                "episodes": len(episodes),
                "transitions": sum(len(ep.times) - 1 for ep in episodes),
                "parquet_sha256": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob("*.parquet"))
                },
                "label": "absolute measured state[t+1]; native delta relative to state[t]; terminal repeat padding",
            },
        )
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.stop()
    print(f"[DATASET] exported immutable joint dataset: {root}")
    return root


def validate_dataset(root: Path):
    root = root.resolve()
    adapter = root / "ur5e_adapter"
    contract = read_contract(adapter / "contract.json")
    ready = json.loads((adapter / "ready.json").read_text())
    if ready["contract_sha256"] != contract_digest(contract):
        raise ValueError("dataset contract changed after export")
    stats_path = adapter / "assets" / STATE_LAYOUT / "norm_stats.json"
    if ready["norm_sha256"] != hashlib.sha256(stats_path.read_bytes()).hexdigest():
        raise ValueError("dataset norm statistics changed after export")
    expected_files = ready["parquet_sha256"]
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*.parquet")}
    if actual_files != set(expected_files):
        raise ValueError("dataset parquet files changed after export")
    for relative, digest in expected_files.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("dataset parquet content changed after export")
    org, name = contract["dataset_id"].split("/")
    if root.parts[-2:] != (org, name):
        raise ValueError("dataset location must end in its declared org/name")
    os.environ["HF_LEROBOT_HOME"] = str(root.parent.parent)
    return contract, ready


def offline_observation(root: Path, run_id: str | None = None, index: int = 0):
    contract, _ = validate_dataset(root)
    run_id = run_id or contract["run_ids"][0]
    if run_id not in contract["run_ids"]:
        raise ValueError("run ID is not in this dataset")
    with np.load(root / "ur5e_adapter" / f"{run_id}.npz", allow_pickle=False) as data:
        vectors = data["vectors"]
        if not 0 <= index < len(vectors) - 1:
            raise ValueError("offline index must select a training observation")
        observation = {
            "state": vectors[index].copy(),
            "prompt": contract["prompt"],
            "images": {
                cam: np.moveaxis(rgb_image(Path(str(data[field][index]))), -1, 0)
                for cam, field in (("cam_high", "head"), ("cam_right_wrist", "wrist"))
            },
        }
        return observation, future_actions(vectors)[index]
