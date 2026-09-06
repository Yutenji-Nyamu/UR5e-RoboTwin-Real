import json
import tempfile
import unittest
from pathlib import Path

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
)
from ur5e_real.operator import resolve_session_reference


class ReplayTest(unittest.TestCase):
    def test_socket_remains_the_default_backend(self):
        cfg = ReplayConfig(Path("actions.csv"), None, "robot.local")
        self.assertEqual(cfg.backend, "socket")

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
