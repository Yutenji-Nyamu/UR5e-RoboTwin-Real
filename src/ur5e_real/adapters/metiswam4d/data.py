"""Small explicit v3-success selection; retain the full task and every gripper transition."""

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ...data.convert_hdf5 import align_nearest
from .contract import FPS, HORIZON, JOINT_ORDER, VERSION, digest, fit_stats, validate_runtime_contract
from ...control.joint import joint_vector


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values(rows, columns):
    result = np.asarray([[float(row[key]) for key in columns] for row in rows], dtype=np.float64)
    if len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("need at least two finite complete samples")
    return result


def _timeline(times):
    gaps = np.diff(times)
    if np.any(gaps <= 0) or np.max(gaps) > 0.25 + 1e-6:
        raise ValueError("nonmonotonic timeline or source gap > 0.25 s")


def prepare(data_root, run_ids, task, output, *, validation_runs=()):
    if not run_ids or len(set(run_ids)) != len(run_ids) or any(not re.fullmatch(r"\d{8}_\d{6}", r) for r in run_ids):
        raise ValueError("select unique explicit run IDs")
    if not set(validation_runs) < set(run_ids) and validation_runs:
        raise ValueError("validation runs must be a subset; at least one training run is required")
    states, actions, valid_steps, images, audit = [], [], [], [], []
    homes, offsets_tcp, initial_grips, all_joints, all_tcp = [], [], [], [], []
    for run in run_ids:
        action_dir = Path(data_root) / "raw" / "action"
        manifest_file = action_dir / f"session_{run}.json"
        meta = json.loads(manifest_file.read_text())
        recording = meta.get("state_recording", {})
        if (meta.get("run_id") != run or meta.get("task") != task or meta.get("schema_version") != 3
                or meta.get("outcome") != "success" or meta.get("recording_status") != "completed"
                or recording.get("joint_source") != "actual_q" or recording.get("joint_position_unit") != "rad"
                or recording.get("joint_order") != ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]):
            raise ValueError(f"{run}: require completed success, selected task, v3 measured joint radians")
        initial = meta.get("initial_robot_state", {})
        homes.append(joint_vector(initial.get("actual_q"), name="initial actual_q"))
        offsets_tcp.append(joint_vector(initial.get("tcp_offset"), name="initial TCP offset"))
        initial_tcp = joint_vector(initial.get("tcp_pose"), name="initial TCP pose")
        initial_grip = meta.get("collection", {}).get("initial_gripper_state")
        if initial_grip not in ("open", "closed", "close"):
            raise ValueError(f"{run}: explicit initial commanded gripper state required")
        initial_grips.append(int(initial_grip != "open"))
        # Resolve conventional raw products under the requested root (portable after a data restore).
        rtde_file = action_dir / f"rtde_tcp_gripper_{run}.csv"
        sync_file = action_dir / f"sync_action_cam_{run}.csv"
        events_file = action_dir / f"gripper_events_{run}.csv"
        rows, sync = _rows(rtde_file), _rows(sync_file)
        t = _values(rows, ["controller_time_s"])[:, 0]
        q = _values(rows, [f"actual_q_{i}" for i in range(6)])
        grip = _values(rows, ["gripper_state"])[:, 0]
        tcp = _values(rows, [f"tcp_{axis}" for axis in ("x", "y", "z")])
        # Envelopes include initialization and all selected demos; normalization below uses train only.
        all_joints.extend([homes[-1][None], q])
        all_tcp.extend([initial_tcp[None, :3], tcp])
        st = _values(sync, ["controller_time_s"])[:, 0]
        _timeline(t)
        _timeline(st)
        if not np.isin(grip, [0, 1]).all() or st[0] < t[0] - 1e-6 or st[-1] > t[-1] + 1e-6:
            raise ValueError(f"{run}: invalid grip labels or image timeline outside robot states")
        grid = st[0] + np.arange(int(np.floor((st[-1] - st[0]) * FPS + 1e-6)) + 1) / FPS
        indices = align_nearest(st, grid)
        offsets = np.abs(st[indices] - grid)
        if offsets.max() > 0.1 + 1e-6:
            raise ValueError(f"{run}: image offset exceeds 0.1 s")
        grid_q = np.stack([np.interp(grid, t, column) for column in q.T], axis=1)
        grid_g = grip[np.maximum(0, np.searchsorted(t, grid + 1e-8, side="right") - 1)]
        vectors = np.column_stack((grid_q, grid_g)).astype(np.float32)
        camera_root = Path(data_root) / "raw" / "camera" / f"cam_dual_{run}"
        for at in range(len(grid) - 1):
            paths = []
            for role in ("head", "wrist"):
                name = sync[indices[at]][f"{role}_image"]
                if Path(name).name != name:
                    raise ValueError("expected a plain frame filename")
                image_path = camera_root / role / name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                paths.append(image_path.relative_to(data_root).as_posix())
            future = at + 1 + np.arange(HORIZON)
            states.append(vectors[at])
            actions.append(vectors[np.minimum(future, len(grid) - 1)])
            valid_steps.append(future < len(grid))
            images.append({"run_id": run, "split": "validation" if run in validation_runs else "train",
                           "controller_time_s": float(grid[at]), "head": paths[0], "wrist": paths[1]})
        audit.append({"run_id": run, "grid_frames": len(grid), "transitions": len(grid) - 1,
                      "max_image_offset_s": float(offsets.max()), "events": _rows(events_file),
                      "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (manifest_file, rtde_file, sync_file, events_file)}})
    state, action, valid = np.stack(states), np.stack(actions), np.stack(valid_steps)
    if (np.max(np.ptp(homes, axis=0)) > 0.03 or not np.allclose(offsets_tcp, offsets_tcp[0], atol=1e-5, rtol=0)
            or len(set(initial_grips)) != 1):
        raise ValueError("selected runs disagree on home posture, TCP offset or initial gripper")
    train = np.asarray([p["split"] == "train" for p in images])
    joints, tcp = np.concatenate(all_joints), np.concatenate(all_tcp)
    contract = {"version": VERSION, "task": task, "fps": FPS, "horizon": HORIZON,
                "slots": list(range(10, 17)), "action": "future measured joint delta + absolute commanded gripper",
                "rgb_layout": "head_above_wrist_each_320x192_RGB", "text": "absent; single task per checkpoint",
                "retiming": "10Hz; linear joints; zero-order grip; nearest image (ties earlier)",
                "crop": "none; full recorded task; no close/open cycle requirement",
                "stats": fit_stats(state[train], action[train], valid[train]),
                "runtime_version": 1, "wire_action": "absolute_joint7", "joint_order": list(JOINT_ORDER),
                "joint_unit": "rad", "home_q": np.mean(homes, axis=0).tolist(),
                "initial_gripper": initial_grips[0], "tcp_offset": offsets_tcp[0].tolist(),
                "joint_lower": (joints.min(axis=0) - 0.10).tolist(), "joint_upper": (joints.max(axis=0) + 0.10).tolist(),
                "tcp_lower": (tcp.min(axis=0) - 0.05).tolist(), "tcp_upper": (tcp.max(axis=0) + 0.05).tolist(),
                "stop": "bounded_chunks", "run_ids": list(run_ids), "validation_runs": list(validation_runs)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "actions.npz", state=state, actions=action, valid=valid)
    (output / "images.json").write_text(json.dumps(images, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    contract["assets_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                 for name in ("actions.npz", "images.json", "audit.json")}
    contract["dataset_id"] = digest(contract)
    validate_runtime_contract(contract)
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "location.json").write_text(json.dumps({"data_root": str(Path(data_root).resolve())}) + "\n")
    return {"episodes": len(run_ids), "training_windows": int(train.sum()), "validation_windows": int((~train).sum()),
            "horizon": HORIZON, "fps": FPS, "dataset_id": contract["dataset_id"]}


def image_paths(dataset, item, *, data_root=None):
    """Relocate raw images without changing the dataset/model contract; accept legacy absolute indices."""
    dataset = Path(dataset)
    if data_root is None and (dataset / "location.json").is_file():
        data_root = json.loads((dataset / "location.json").read_text())["data_root"]
    paths = []
    for role in ("head", "wrist"):
        path = Path(item[role])
        if not path.is_absolute():
            if data_root is None or ".." in path.parts:
                raise ValueError("relative image index requires a data root and must stay inside it")
            path = Path(data_root) / path
        paths.append(path)
    return paths


def read_dataset(dataset):
    dataset = Path(dataset)
    contract = validate_runtime_contract(json.loads((dataset / "contract.json").read_text()))
    identity = dict(contract)
    if identity.pop("dataset_id") != digest(identity):
        raise ValueError("dataset contract identity mismatch")
    for name in ("actions.npz", "images.json", "audit.json"):
        if hashlib.sha256((dataset / name).read_bytes()).hexdigest() != contract["assets_sha256"][name]:
            raise ValueError(f"dataset asset differs: {name}")
    with np.load(dataset / "actions.npz", allow_pickle=False) as values:
        state, action, valid = values["state"], values["actions"], values["valid"]
    images = json.loads((dataset / "images.json").read_text())
    n = len(images)
    if (not n or state.shape != (n, 7) or action.shape != (n, HORIZON, 7) or valid.shape != (n, HORIZON)
            or not np.isfinite(state).all() or not np.isfinite(action).all() or valid.dtype != np.bool_):
        raise ValueError("dataset shapes/values are invalid")
    return contract, state, action, valid, images


def read_rgb(path):
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
