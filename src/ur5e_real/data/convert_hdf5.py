from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .schema import CAMERA_GROUP_MAP, SCHEMA_VERSION, tcp_gripper_vectors


@dataclass(frozen=True)
class RawSession:
    run_id: str
    action_csv: Path
    camera_dir: Path
    sync_csv: Path | None


def discover_sessions(
    action_dir: Path,
    camera_dir: Path,
    *,
    task_name: str | None = None,
    include_unreviewed: bool = False,
) -> list[RawSession]:
    sessions: list[RawSession] = []
    if not action_dir.is_dir() or not camera_dir.is_dir():
        return sessions
    for action_csv in action_dir.glob("rtde_tcp_gripper_*.csv"):
        run_id = action_csv.stem.removeprefix("rtde_tcp_gripper_")
        manifest_path = action_dir / f"session_{run_id}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if manifest is None:
            if not include_unreviewed:
                continue
        else:
            if task_name is not None and manifest.get("task") != task_name:
                continue
            if not include_unreviewed and manifest.get("outcome") != "success":
                continue
        camera_run = camera_dir / f"cam_dual_{run_id}"
        if not camera_run.is_dir():
            continue
        sync_path = action_dir / f"sync_action_cam_{run_id}.csv"
        sessions.append(RawSession(run_id, action_csv, camera_run, sync_path if sync_path.is_file() else None))
    return sorted(sessions, key=lambda item: item.run_id)


def load_actions(path: Path):
    import numpy as np

    times: list[float] = []
    poses: list[list[float]] = []
    gripper: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ["controller_time_s", "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing action columns {missing}")
        for row in reader:
            times.append(float(row["controller_time_s"]))
            poses.append([float(row[name]) for name in required[1:]])
            gripper.append(float(row.get("gripper_state") or 0.0))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(poses, dtype=np.float32),
        np.asarray(gripper, dtype=np.float32),
    )


def load_sync(path: Path):
    import numpy as np

    times: list[float] = []
    head: list[str] = []
    wrist: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["controller_time_s"]))
            head.append(row["head_image"])
            wrist.append(row["wrist_image"])
    return np.asarray(times, dtype=np.float64), head, wrist


def align_nearest(action_times, query_times):
    import numpy as np

    if len(action_times) == 0:
        raise ValueError("cannot align against an empty action timeline")
    right = np.searchsorted(action_times, query_times, side="left")
    right = np.clip(right, 0, len(action_times) - 1)
    left = np.clip(right - 1, 0, len(action_times) - 1)
    choose_left = np.abs(query_times - action_times[left]) <= np.abs(query_times - action_times[right])
    return np.where(choose_left, left, right).astype(np.int64)


def _encode_jpeg(path: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"failed to encode image: {path}")
    return np.frombuffer(encoded.tobytes(), dtype=np.uint8)


def write_episode(path: Path, poses, gripper, head_paths: list[Path], wrist_paths: list[Path], run_id: str) -> None:
    import h5py
    import numpy as np

    length = len(head_paths)
    if not (length == len(wrist_paths) == len(poses) == len(gripper)):
        raise ValueError("episode arrays have different lengths")
    if length == 0:
        raise ValueError(f"session {run_id} contains no aligned samples")
    vectors = tcp_gripper_vectors(poses, gripper)
    poses = vectors[:, :6]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["source_run_id"] = run_id
        action = handle.create_group("joint_action")
        action.create_dataset("left_arm", data=poses)
        action.create_dataset("left_gripper", data=np.zeros(length, dtype=np.float32))
        action.create_dataset("right_arm", data=poses)
        action.create_dataset("right_gripper", data=gripper)
        action.create_dataset("vector", data=vectors)

        observation = handle.create_group("observation")
        encoded_type = h5py.vlen_dtype(np.dtype("uint8"))
        datasets = {
            name: observation.create_group(name).create_dataset("rgb", (length,), dtype=encoded_type)
            for name in CAMERA_GROUP_MAP
        }
        for index, (head_path, wrist_path) in enumerate(zip(head_paths, wrist_paths)):
            encoded = {"head": _encode_jpeg(head_path), "wrist": _encode_jpeg(wrist_path)}
            for camera_name, source in CAMERA_GROUP_MAP.items():
                datasets[camera_name][index] = encoded[source]


def _aligned_session(session: RawSession):
    import numpy as np

    action_times, poses, gripper = load_actions(session.action_csv)
    head_dir = session.camera_dir / "head"
    wrist_dir = session.camera_dir / "wrist"
    if session.sync_csv is not None:
        sync_times, head_names, wrist_names = load_sync(session.sync_csv)
        triples = [
            (time_value, head_dir / head_name, wrist_dir / wrist_name)
            for time_value, head_name, wrist_name in zip(sync_times, head_names, wrist_names)
            if (head_dir / head_name).is_file() and (wrist_dir / wrist_name).is_file()
        ]
        if not triples:
            return poses[:0], gripper[:0], [], []
        valid_times = np.asarray([item[0] for item in triples], dtype=np.float64)
        indices = align_nearest(action_times, valid_times)
        return poses[indices], gripper[indices], [item[1] for item in triples], [item[2] for item in triples]

    head_paths = sorted(head_dir.glob("frame_*.png"))
    wrist_paths = sorted(wrist_dir.glob("frame_*.png"))
    length = min(len(poses), len(head_paths), len(wrist_paths))
    return poses[:length], gripper[:length], head_paths[:length], wrist_paths[:length]


def convert_raw_sessions(
    action_dir: Path,
    camera_dir: Path,
    output_root: Path,
    task_name: str,
    task_config: str,
    *,
    include_unreviewed: bool = False,
) -> Path:
    sessions = discover_sessions(
        action_dir,
        camera_dir,
        task_name=task_name,
        include_unreviewed=include_unreviewed,
    )
    if not sessions:
        qualifier = "matched" if include_unreviewed else "reviewed-success"
        raise RuntimeError(f"no {qualifier} raw sessions in {action_dir} and {camera_dir}")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"run_{run_tag}"
    episode_dir = run_root / task_name / task_config / "data"
    for index, session in enumerate(sessions):
        poses, gripper, head_paths, wrist_paths = _aligned_session(session)
        output_path = episode_dir / f"episode{index}.hdf5"
        write_episode(output_path, poses, gripper, head_paths, wrist_paths, session.run_id)
        print(f"[SAVED] {output_path}")
    return run_root
