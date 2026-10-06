import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from ur5e_real.replay import (
    ReplayConfig,
    apply_rtde_replay_gripper,
    build_rtde_chunks,
    build_segment_program,
    build_segments,
    filter_waypoints,
    load_action_rows,
    resolve_replay_paths,
    smooth_rotation_vectors,
    run_replay,
)
from ur5e_real.operator import resolve_session_reference


class ReplayTest(unittest.TestCase):
    def test_rtde_is_the_default_backend_with_80_ms_inference_hold(self):
        cfg = ReplayConfig(Path("actions.csv"), None, "robot.local")
        self.assertEqual(cfg.backend, "rtde")
        self.assertEqual(cfg.chunk_gap_s, 0.08)

    def test_latest_session_uses_newest_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            action_dir = Path(directory)
            older = action_dir / "session_20260903_120000.json"
            latest = action_dir / "session_20260905_180000.json"
            older.touch()
            latest.touch()
            self.assertEqual(resolve_session_reference("latest", action_dir), latest.resolve())

    def test_load_filter_and_program(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.csv"
            path.write_text(
                "controller_time_s,tcp_x,tcp_y,tcp_z,tcp_rx,tcp_ry,tcp_rz\n"
                "0,0,0,0,0,0,0\n"
                "1,0.0001,0,0,0,0,0\n"
                "2,0.02,0,0,0,0,0\n",
                encoding="utf-8",
            )
            rows = load_action_rows(path)
            self.assertIsNone(rows[0]["gripper_state"])
            self.assertEqual(smooth_rotation_vectors(rows), 0)
            filtered = filter_waypoints(rows, [], 0.003, 0.03)
            self.assertEqual(len(filtered), 2)
            segment = build_segments(filtered, [])[0]
            cfg = ReplayConfig(path, None, "robot.local")
            program, wait = build_segment_program(filtered, segment, 0, cfg)
            self.assertIn("def ur5e_replay_segment_000", program)
            self.assertIn("movel(p[", program)
            self.assertGreaterEqual(wait, 0.5)

    def test_session_manifest_resolves_replay_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "session_test.json"
            manifest.write_text(
                json.dumps(
                    {
                        "run_id": "test",
                        "paths": {
                            "rtde": "rtde.csv",
                            "gripper_events": "events.csv",
                        },
                    }
                ),
                encoding="utf-8",
            )
            action, events = resolve_replay_paths(manifest)
            self.assertEqual(action, root / "rtde.csv")
            self.assertEqual(events, root / "events.csv")

    def test_rtde_replay_groups_six_actions_and_maps_events_to_waypoints(self):
        rows = [
            {"row_index": index, "controller_time_s": index / 10, "pose": [index, 0, 0, 0, 0, 0]}
            for index in range(14)
        ]
        events = [
            {"controller_time_s": 0.15, "event": "close"},
            {"controller_time_s": 0.75, "event": "open"},
        ]
        chunks = build_rtde_chunks(rows, events)
        self.assertEqual([len(chunk) for chunk in chunks], [6, 6, 1])
        self.assertEqual(chunks[0][1]["events"][0]["event"], "close")
        self.assertEqual(chunks[1][1]["events"][0]["event"], "open")
        self.assertEqual(len(build_rtde_chunks(rows, events, max_chunks=1)), 1)

    def test_rtde_replay_preserves_repeated_close_button_presses(self):
        class FakeGripper:
            def __init__(self):
                self.commands = []

            def close(self):
                self.commands.append("close")

            def open(self):
                self.commands.append("open")

        gripper = FakeGripper()
        for event in ("close", "close", "open"):
            apply_rtde_replay_gripper(
                {
                    "row": {"gripper_state": 1.0 if event == "close" else 0.0},
                    "events": [{"event": event}],
                },
                gripper,
                None,
                use_recorded_events=True,
            )
        self.assertEqual(gripper.commands, ["close", "close", "open"])


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("flags,backend,seconds", [([], "rtde", 0.08),
    (["--inference-ms", "250"], "rtde", 0.25), (["--inference-ms", "0"], "rtde", 0),
    (["--chunk-gap", "0.125"], "rtde", 0.125), (["--backend", "socket"], "socket", 0.08)])
def test_both_replay_cli_entries_forward_delay(flags, backend, seconds):
    from ur5e_real import operator, cli
    from ur5e_real.config import load_config

    with patch("ur5e_real.config.validate_config"):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/lab.example.yaml")
    with patch.object(operator, "_enter_repository"), patch.object(operator, "load_config", return_value=cfg), \
            patch.object(operator, "_session_path", return_value=Path("session.json")), \
            patch("ur5e_real.replay.resolve_replay_paths", return_value=(Path("actions.csv"), None)), \
            patch("ur5e_real.replay.run_replay") as run, \
            patch("sys.argv", ["ur5e-replay", "latest", *flags]):
        operator.replay()
        config = run.call_args.args[0]
        assert config.backend == backend and config.chunk_gap_s == seconds
    with patch("ur5e_real.cli.load_config", return_value=cfg), \
            patch("ur5e_real.replay.resolve_replay_paths", return_value=(Path("actions.csv"), None)), \
            patch("ur5e_real.replay.run_replay") as run:
        cli.main(["replay", "--config", "unused.yaml", "session.json", *flags])
        config = run.call_args.args[0]
        assert config.backend == backend and config.chunk_gap_s == seconds


@pytest.mark.parametrize("delay", [-0.001, float("nan"), float("inf")])
def test_invalid_delay_fails_before_hardware_access(delay):
    cfg = ReplayConfig(Path("absent.csv"), None, "unused", execute=True, chunk_gap_s=delay)
    with patch("ur5e_real.replay.ServoJController") as controller:
        with pytest.raises(ValueError, match="finite and non-negative"):
            run_replay(cfg)
        controller.assert_not_called()


def test_rtde_delay_is_applied_only_between_chunks():
    rows = [{"pose": [0.0] * 6, "controller_time_s": i / 10, "gripper_state": 0} for i in range(14)]
    cfg = ReplayConfig(Path("unused.csv"), None, "unused", execute=True, go_to_start=False,
                       enable_gripper=False, chunk_gap_s=0.25,
                       servoj_config_xml=Path("test.xml"), servoj_program_script=Path("test.script"))
    with patch("ur5e_real.replay.load_action_rows", return_value=rows), \
            patch("ur5e_real.replay.ServoJController") as controller, \
            patch("ur5e_real.replay.start_servoj_program"), \
            patch("ur5e_real.replay.stream_tcp_chunk") as stream, \
            patch("ur5e_real.replay.time.sleep") as sleep:
        run_replay(cfg)
        assert stream.call_count == 3
        assert [c.args for c in sleep.call_args_list] == [(0.25,), (0.25,)]
        controller.return_value.stop.assert_called_once()


def test_rtde_does_not_silently_ignore_old_socket_segment_limit():
    with pytest.raises(ValueError, match="--chunks"):
        run_replay(ReplayConfig(Path("unused.csv"), None, "unused", max_segments=1))


def test_delay_units_cannot_be_combined():
    from ur5e_real.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["replay", "--config", "unused", "unused.csv",
                             "--inference-ms", "100", "--chunk-gap", "0.1"])
