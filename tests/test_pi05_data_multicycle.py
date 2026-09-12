import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ur5e_real.adapters.robotwin_pi05.contract import JOINT_ORDER
from ur5e_real.adapters.robotwin_pi05.native import REPOSITORY
from ur5e_real.adapters.robotwin_pi05.process_data import audit_selection, load_selection, read_episode


@pytest.fixture
def two_cycle_episode(tmp_path):
    selection = load_selection(REPOSITORY / "configs/pi05_stack_blocks_three_joint_5_20260911.json")
    run = selection["run_ids"][0]
    selection["run_ids"] = [run]
    action_dir = tmp_path / "raw/action"
    action_dir.mkdir(parents=True)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame.png").touch()
    times = 100 + np.arange(201) / 10
    joint = np.interp(times, [100, 103, 105, 107, 110, 112, 114, 116, 120], [0, 0, 0.1, 0.1, 0.1, 0.2, 0.2, 0, 0])
    grip = (((times >= 104) & (times < 107)) | ((times >= 111) & (times < 115))).astype(int)

    def write_csv(name, fields, rows):
        path = action_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return str(path)

    paths = {
        "rtde": write_csv(
            "rtde.csv",
            [
                "controller_time_s",
                *[f"actual_q_{i}" for i in range(6)],
                *[f"tcp_{key}" for key in ("x", "y", "z", "rx", "ry", "rz")],
                "gripper_state",
            ],
            [[t, q, 0, 0, 0, 0, 0, q, 0, 0, 0, 0, 0, g] for t, q, g in zip(times, joint, grip)],
        ),
        "sync": write_csv(
            "sync.csv",
            ["controller_time_s", "head_image", "wrist_image"],
            [[t, "frame.png", "frame.png"] for t in times],
        ),
        "gripper_events": write_csv(
            "events.csv", ["controller_time_s", "event"], [[104, "close"], [107, "open"], [111, "close"], [115, "open"]]
        ),
        "head_frames": str(image_dir),
        "wrist_frames": str(image_dir),
    }
    manifest = {
        "run_id": run,
        "schema_version": 3,
        "task": selection["task"],
        "outcome": "success",
        "recording_status": "completed",
        "paths": paths,
        "state_recording": {
            "joint_source": "actual_q",
            "joint_position_unit": "rad",
            "joint_order": list(JOINT_ORDER),
            "tcp_source": "actual_TCP_pose",
        },
        "initial_robot_state": {"actual_q": [0] * 6, "tcp_offset": [0] * 6},
    }
    (action_dir / f"session_{run}.json").write_text(json.dumps(manifest))
    return tmp_path, selection, paths


def test_two_cycles_preserve_middle_pause_and_final_release_tail(two_cycle_episode):
    root, selection, _ = two_cycle_episode
    episodes, report = audit_selection(root, selection)
    episode = episodes[0]
    assert np.count_nonzero(np.diff(episode.gripper) > 0) == 2
    assert np.count_nonzero(np.diff(episode.gripper) < 0) == 2
    assert np.count_nonzero((episode.times >= 107) & (episode.times < 110)) >= 29
    assert episode.audit["last_observation_release_tail_s"] >= 1.0
    assert episode.audit["trimmed_lead_s"] > 0 and episode.audit["trimmed_tail_s"] > 0
    assert report["contract"]["gripper_cycles"] == 2


def test_legacy_selection_still_requires_one_cycle(two_cycle_episode):
    root, selection, _ = two_cycle_episode
    selection.pop("gripper_cycles")
    with pytest.raises(ValueError, match="exactly 1"):
        read_episode(root, selection["run_ids"][0], selection)


def test_later_event_timestamps_are_checked(two_cycle_episode):
    root, selection, paths = two_cycle_episode
    from pathlib import Path

    Path(paths["gripper_events"]).write_text("controller_time_s,event\n104,close\n107,open\n115,close\n111,open\n")
    with pytest.raises(ValueError, match="gripper events"):
        read_episode(root, selection["run_ids"][0], selection)


@pytest.mark.parametrize("cycles", [0, -1, 1.5, True])
def test_invalid_cycle_selection_rejected(tmp_path, cycles):
    selection = json.loads((REPOSITORY / "configs/pi05_cube_joint_5.json").read_text())
    selection["gripper_cycles"] = cycles
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="gripper_cycles"):
        load_selection(path)


REPEATED_COMMANDS = (
    "controller_time_s,event\n104,close\n105,close\n107,open\n109,open\n111,close\n112,close\n115,open\n118,open\n"
)


def test_repeated_commands_require_explicit_selection(two_cycle_episode):
    root, selection, paths = two_cycle_episode
    Path(paths["gripper_events"]).write_text(REPEATED_COMMANDS)
    with pytest.raises(ValueError, match="exactly 2"):
        read_episode(root, selection["run_ids"][0], selection)


def test_repeated_commands_preserve_raw_events_labels_and_last_open_tail(two_cycle_episode):
    root, selection, paths = two_cycle_episode
    selection["allow_repeated_gripper_commands"] = True
    events = Path(paths["gripper_events"])
    events.write_text(REPEATED_COMMANDS)
    source_before = {key: Path(paths[key]).read_bytes() for key in ("rtde", "sync", "gripper_events")}
    episode = read_episode(root, selection["run_ids"][0], selection)
    expected = (
        ((episode.times >= 104) & (episode.times < 107)) | ((episode.times >= 111) & (episode.times < 115))
    ).astype(int)
    np.testing.assert_array_equal(episode.gripper, expected)
    assert len(episode.audit["gripper_events"]) == 8
    assert episode.audit["repeated_gripper_commands"] == 4
    assert [row["controller_time_s"] for row in episode.audit["binary_gripper_events"]] == [104, 107, 111, 115]
    assert episode.audit["gripper_events"][-1]["controller_time_s"] == 118
    assert episode.times[-2] >= 119  # Final command, not the earlier binary state transition.
    assert episode.audit["last_observation_release_tail_s"] >= 1.0
    for key, original in source_before.items():
        assert Path(paths[key]).read_bytes() == original


@pytest.mark.parametrize("invalid", ["103", "104", "nan"])
def test_repeated_event_timestamps_are_not_skipped(two_cycle_episode, invalid):
    root, selection, paths = two_cycle_episode
    selection["allow_repeated_gripper_commands"] = True
    Path(paths["gripper_events"]).write_text(REPEATED_COMMANDS.replace("105,close", f"{invalid},close"))
    with pytest.raises(ValueError, match="gripper events"):
        read_episode(root, selection["run_ids"][0], selection)


@pytest.mark.parametrize("events", ["104,open\n115,close", "104,close\n105,close", "104,stop\n115,open"])
def test_repeat_opt_in_does_not_allow_bad_cycles(two_cycle_episode, events):
    root, selection, paths = two_cycle_episode
    selection.update(gripper_cycles=1, allow_repeated_gripper_commands=True)
    Path(paths["gripper_events"]).write_text(f"controller_time_s,event\n{events}\n")
    with pytest.raises(ValueError, match="exactly 1"):
        read_episode(root, selection["run_ids"][0], selection)


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_repeat_opt_in_must_be_boolean(tmp_path, flag):
    selection = json.loads((REPOSITORY / "configs/pi05_cube_joint_5.json").read_text())
    selection["allow_repeated_gripper_commands"] = flag
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="allow_repeated_gripper_commands"):
        load_selection(path)
