"""Small explicit v3-success selection; retain the full task and every gripper transition."""

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ...data.convert_hdf5 import align_nearest
from .contract import FPS, HORIZON, VERSION, fit_stats


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


def prepare(data_root, run_ids, task, output):
    if not run_ids or len(set(run_ids)) != len(run_ids) or any(not re.fullmatch(r"\d{8}_\d{6}", r) for r in run_ids):
        raise ValueError("select unique explicit run IDs")
    states, actions, valid_steps, images, audit = [], [], [], [], []
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
        # Resolve conventional raw products under the requested root (portable after a data restore).
        rtde_file = action_dir / f"rtde_tcp_gripper_{run}.csv"
        sync_file = action_dir / f"sync_action_cam_{run}.csv"
        events_file = action_dir / f"gripper_events_{run}.csv"
        rows, sync = _rows(rtde_file), _rows(sync_file)
        t = _values(rows, ["controller_time_s"])[:, 0]
        q = _values(rows, [f"actual_q_{i}" for i in range(6)])
        grip = _values(rows, ["gripper_state"])[:, 0]
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
                paths.append(str(image_path.resolve()))
            future = at + 1 + np.arange(HORIZON)
            states.append(vectors[at])
            actions.append(vectors[np.minimum(future, len(grid) - 1)])
            valid_steps.append(future < len(grid))
            images.append({"run_id": run, "controller_time_s": float(grid[at]), "head": paths[0], "wrist": paths[1]})
        audit.append({"run_id": run, "grid_frames": len(grid), "transitions": len(grid) - 1,
                      "max_image_offset_s": float(offsets.max()), "events": _rows(events_file),
                      "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (manifest_file, rtde_file, sync_file, events_file)}})
    state, action, valid = np.stack(states), np.stack(actions), np.stack(valid_steps)
    contract = {"version": VERSION, "task": task, "fps": FPS, "horizon": HORIZON,
                "slots": list(range(10, 17)), "action": "future measured joint delta + absolute commanded gripper",
                "rgb_layout": "head_above_wrist_each_320x192_RGB", "text": "absent; single task per checkpoint",
                "retiming": "10Hz; linear joints; zero-order grip; nearest image (ties earlier)",
                "crop": "none; full recorded task; no close/open cycle requirement", "stats": fit_stats(state, action, valid)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "actions.npz", state=state, actions=action, valid=valid)
    (output / "images.json").write_text(json.dumps(images, indent=2) + "\n")
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {"episodes": len(run_ids), "training_windows": len(state), "horizon": HORIZON, "fps": FPS}


def read_rgb(path):
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
