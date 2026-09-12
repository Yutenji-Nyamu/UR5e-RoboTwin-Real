"""Read v3 demonstrations, explicitly retime to 10 Hz, export native LeRobot data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ...data.convert_hdf5 import align_nearest
from .contract import JOINT_ORDER, NATIVE_PATCH, ROBOTWIN_COMMIT, STATE_LAYOUT, encode_state, validate_contract


@dataclass
class JointEpisode:
    run_id: str
    times: np.ndarray
    q: np.ndarray
    tcp: np.ndarray
    gripper: np.ndarray
    head: list[Path]
    wrist: list[Path]
    initial_q: np.ndarray
    tcp_offset: np.ndarray
    audit: dict

    @property
    def vectors(self):
        return np.stack([encode_state(q, grip) for q, grip in zip(self.q, self.gripper)])


def load_selection(path: Path) -> dict:
    selection = json.loads(path.read_text(encoding="utf-8"))
    ids = selection["run_ids"]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"\d{8}_\d{6}", run) for run in ids):
        raise ValueError("selection must contain unique explicit run IDs")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", selection["repo_id"]):
        raise ValueError("dataset repo_id must be a safe org/name identifier")
    if selection["fps"] != 10 or selection["action_horizon"] != 50:
        raise ValueError("this UR pi05 dataset contract uses 10 Hz / horizon 50")
    for name in (
        "motion_threshold_rad",
        "release_tail_s",
        "max_source_gap_s",
        "max_image_offset_s",
        "joint_margin_rad",
        "tcp_margin_m",
    ):
        if not np.isfinite(selection[name]) or selection[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if selection["release_tail_s"] < 1:
        raise ValueError("release tail must retain at least one second")
    for name in ("lead_in_steps", "tail_steps"):
        if not isinstance(selection[name], int) or selection[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not selection.get("prompt") or not selection.get("task"):
        raise ValueError("task and prompt are required")
    if type(selection.get("allow_repeated_gripper_commands", False)) is not bool:
        raise ValueError("allow_repeated_gripper_commands must be a boolean")
    gripper_cycles(selection)
    return selection


def gripper_cycles(selection: dict) -> int:
    cycles = selection.get("gripper_cycles", 1)
    if type(cycles) is not int or cycles < 1:
        raise ValueError("gripper_cycles must be a positive integer")
    return cycles


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matrix(rows, columns, label):
    try:
        result = np.asarray([[float(row[key]) for key in columns] for row in rows], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"{label}: missing or invalid fields") from exc
    if len(rows) < 2 or result.shape != (len(rows), len(columns)) or not np.isfinite(result).all():
        raise ValueError(f"{label}: need at least two finite complete samples")
    return result


def _timeline(times, maximum_gap: float, label: str):
    if not np.isfinite(times).all() or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"{label}: timestamps must be finite and strictly increasing")
    if np.max(np.diff(times)) > maximum_gap + 1e-6:
        raise ValueError(f"{label}: source gap exceeds the explicit selection tolerance")


def crop_bounds(q, grip, threshold: float, lead: int, tail: int):
    from_start = (np.max(np.abs(q - q[0]), axis=1) > threshold) | (grip != grip[0])
    from_end = (np.max(np.abs(q - q[-1]), axis=1) > threshold) | (grip != grip[-1])
    if not np.any(from_start) or not np.any(from_end):
        raise ValueError("demonstration has no measurable activity")
    return max(0, int(np.flatnonzero(from_start)[0]) - lead), min(len(q), int(np.flatnonzero(from_end)[-1]) + tail + 1)


def read_episode(data_root: Path, run_id: str, selection: dict) -> JointEpisode:
    action_dir = data_root / "raw" / "action"
    manifest_path = action_dir / f"session_{run_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = manifest.get("state_recording") or {}
    if (
        manifest.get("run_id") != run_id
        or manifest.get("schema_version") != 3
        or manifest.get("task") != selection["task"]
        or manifest.get("outcome") != "success"
        or manifest.get("recording_status") != "completed"
    ):
        raise ValueError(f"{run_id}: require a completed reviewed-success v3 episode of the selected task")
    if (
        recording.get("joint_source") != "actual_q"
        or recording.get("joint_position_unit") != "rad"
        or recording.get("joint_order") != list(JOINT_ORDER)
        or recording.get("tcp_source") != "actual_TCP_pose"
    ):
        raise ValueError(f"{run_id}: measured joint/TCP semantics do not match")
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    rows, sync, events = _csv(paths["rtde"]), _csv(paths["sync"]), _csv(paths["gripper_events"])
    t = _matrix(rows, ["controller_time_s"], run_id).ravel()
    q = _matrix(rows, [f"actual_q_{i}" for i in range(6)], run_id)
    tcp = _matrix(rows, [f"tcp_{key}" for key in ("x", "y", "z", "rx", "ry", "rz")], run_id)
    grip = _matrix(rows, ["gripper_state"], run_id).ravel()
    if not np.isin(grip, [0, 1]).all():
        raise ValueError(f"{run_id}: raw gripper labels must be binary commands")
    st = _matrix(sync, ["controller_time_s"], f"{run_id} images").ravel()
    _timeline(t, selection["max_source_gap_s"], run_id)
    _timeline(st, selection["max_source_gap_s"], f"{run_id} images")
    if st[0] < t[0] - 1e-6 or st[-1] > t[-1] + 1e-6:
        raise ValueError(f"{run_id}: image timestamps outside the robot state timeline")
    cycles = gripper_cycles(selection)
    allow_repeats = selection.get("allow_repeated_gripper_commands", False)
    binary_events = [event for i, event in enumerate(events) if i == 0 or event["event"] != events[i - 1]["event"]]
    # This opt-in validates the existing binary action labels, not button-press replay.
    # Keep every source event and timestamp for provenance and final-release cropping.
    checked_events = binary_events if allow_repeats else events
    if [event["event"] for event in checked_events] != ["close", "open"] * cycles:
        raise ValueError(f"{run_id}: require exactly {cycles} ordered close/open gripper cycles")
    event_times = np.asarray([float(event["controller_time_s"]) for event in events])
    if (
        not np.isfinite(event_times).all()
        or np.any(np.diff(event_times) <= 0)
        or event_times[0] < st[0]
        or event_times[-1] > st[-1]
    ):
        raise ValueError(f"{run_id}: invalid or out-of-range gripper events")
    head, wrist = [], []
    for item in sync:
        for name, target in (("head", head), ("wrist", wrist)):
            image_path = paths[f"{name}_frames"] / item[f"{name}_image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            target.append(image_path)
    sync_q = np.stack([np.interp(st, t, column) for column in q.T], axis=1)
    sync_grip = grip[np.maximum(0, np.searchsorted(t, st + 1e-8, side="right") - 1)]
    start, stop = crop_bounds(
        sync_q, sync_grip, selection["motion_threshold_rad"], selection["lead_in_steps"], selection["tail_steps"]
    )
    start = min(start, int(np.searchsorted(st, event_times[0], side="right")) - 1)
    # The last grid point supplies the next-state label, not a training observation.
    # Retain one extra tick so even the last observed image covers open + 1 second.
    required_end = event_times[-1] + selection["release_tail_s"] + 1 / selection["fps"]
    if st[-1] < required_end - 1e-6:
        raise ValueError(f"{run_id}: raw images do not cover the required release tail")
    end_time = max(st[stop - 1], required_end)
    steps = int(np.ceil((end_time - st[start]) * selection["fps"] - 1e-6))
    grid = st[start] + np.arange(steps + 1) / selection["fps"]
    grid = grid[grid <= st[-1] + 1e-6]
    if len(grid) < 2 or grid[-1] < required_end - 1e-6:
        raise ValueError(f"{run_id}: insufficient tail after 10 Hz retiming")
    image_indices = align_nearest(st, grid)
    offsets = st[image_indices] - grid
    if np.max(np.abs(offsets)) > selection["max_image_offset_s"] + 1e-6:
        raise ValueError(f"{run_id}: image age exceeds explicit retiming tolerance")
    grid_q = np.stack([np.interp(grid, t, column) for column in q.T], axis=1)
    grid_tcp = np.stack([np.interp(grid, t, column) for column in tcp.T], axis=1)
    grid_grip = grip[np.maximum(0, np.searchsorted(t, grid + 1e-8, side="right") - 1)]
    initial = manifest["initial_robot_state"]
    audit = {
        "run_id": run_id,
        "raw_schema_version": 3,
        "raw_robot_samples": len(rows),
        "raw_frame_pairs": len(sync),
        "raw_image_start_s": float(st[0]),
        "raw_image_stop_s": float(st[-1]),
        "trimmed_lead_s": float(grid[0] - st[0]),
        "trimmed_tail_s": float(st[-1] - grid[-1]),
        "gripper_cycles": cycles,
        "gripper_events": [
            {"event": event["event"], "controller_time_s": float(at)} for event, at in zip(events, event_times)
        ],
        **(
            {
                "gripper_event_validation": "binary_transitions_allow_repeated_commands",
                "repeated_gripper_commands": len(events) - len(binary_events),
                "binary_gripper_events": [
                    {"event": event["event"], "controller_time_s": float(event["controller_time_s"])}
                    for event in binary_events
                ],
                "gripper_label_scope": "binary_command_state; repeated presses are not separate action labels",
            }
            if allow_repeats
            else {}
        ),
        "crop_start": start,
        "motion_crop_stop_exclusive": stop,
        "grid_frames": len(grid),
        "training_transitions": len(grid) - 1,
        "grid_controller_start_s": float(grid[0]),
        "grid_controller_stop_s": float(grid[-1]),
        "release_tail_s": float(grid[-1] - event_times[-1]),
        "last_observation_release_tail_s": float(grid[-2] - event_times[-1]),
        "source_image_gaps": [
            {"after_index": int(i), "dt_s": float(dt)} for i, dt in enumerate(np.diff(st)) if dt > 0.15
        ],
        "reused_image_frames": int(np.count_nonzero(np.diff(image_indices) == 0)),
        "max_image_offset_s": float(np.max(np.abs(offsets))),
        "resampling": "10Hz; linear measured joints/TCP; zero-order gripper; nearest image, ties prefer earlier",
        "joint_velocity_max_rad_s": np.max(np.abs(np.diff(grid_q, axis=0)) * selection["fps"], axis=0).tolist(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (manifest_path, paths["rtde"], paths["sync"], paths["gripper_events"])
        },
    }
    return JointEpisode(
        run_id,
        grid,
        grid_q,
        grid_tcp,
        grid_grip,
        [head[i] for i in image_indices],
        [wrist[i] for i in image_indices],
        np.asarray(initial["actual_q"]),
        np.asarray(initial["tcp_offset"]),
        audit,
    )


def build_contract(episodes: list[JointEpisode], selection: dict) -> dict:
    starts = np.stack([episode.initial_q for episode in episodes])
    offsets = np.stack([episode.tcp_offset for episode in episodes])
    if np.max(np.ptp(starts, axis=0)) > 0.03 or not np.allclose(offsets, offsets[0], atol=1e-5, rtol=0):
        raise ValueError("demonstrations disagree on home joints or TCP offset")
    q = np.concatenate([episode.q for episode in episodes] + [starts])
    tcp = np.concatenate([episode.tcp[:, :3] for episode in episodes])
    return validate_contract(
        {
            "version": 1,
            "policy": "pi05",
            "action_space": "joint_position",
            "state_layout": STATE_LAYOUT,
            "joint_unit": "rad",
            "joint_order": list(JOINT_ORDER),
            "fps": selection["fps"],
            "action_horizon": selection["action_horizon"],
            "adapt_to_pi": False,
            "delta_joint_actions": True,
            "robotwin_commit": ROBOTWIN_COMMIT,
            "dataset_id": selection["repo_id"],
            "prompt": selection["prompt"],
            "native_patch": NATIVE_PATCH,
            "run_ids": selection["run_ids"],
            **({"gripper_cycles": gripper_cycles(selection)} if "gripper_cycles" in selection else {}),
            "home_q": np.mean(starts, axis=0).tolist(),
            "joint_lower": (np.min(q, axis=0) - selection["joint_margin_rad"]).tolist(),
            "joint_upper": (np.max(q, axis=0) + selection["joint_margin_rad"]).tolist(),
            "tcp_lower": (np.min(tcp, axis=0) - selection["tcp_margin_m"]).tolist(),
            "tcp_upper": (np.max(tcp, axis=0) + selection["tcp_margin_m"]).tolist(),
            "tcp_offset": offsets[0].tolist(),
            "gripper_semantics": "commanded_open_0_closed_1",
            "image_semantics": "head_and_right_wrist_RGB; left_wrist_missing_mask_false",
        }
    )


def audit_selection(data_root: Path, selection: dict):
    episodes = [read_episode(data_root, run_id, selection) for run_id in selection["run_ids"]]
    return episodes, {
        "selection": selection,
        "contract": build_contract(episodes, selection),
        "episodes": [episode.audit for episode in episodes],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--lerobot-home", type=Path, help="export a new immutable dataset here; omit for audit only")
    args = parser.parse_args()
    episodes, report = audit_selection(args.data_root, load_selection(args.selection))
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    if args.lerobot_home:
        from .dataset import export_dataset

        export_dataset(episodes, report, args.lerobot_home)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
