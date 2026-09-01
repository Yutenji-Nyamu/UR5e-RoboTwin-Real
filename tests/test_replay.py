import tempfile
import unittest
from pathlib import Path

from ur5e_real.replay import (
    ReplayConfig,
    build_segment_program,
    build_segments,
    filter_waypoints,
    load_action_rows,
    smooth_rotation_vectors,
)


class ReplayTest(unittest.TestCase):
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
            self.assertEqual(smooth_rotation_vectors(rows), 0)
            filtered = filter_waypoints(rows, [], 0.003, 0.03)
            self.assertEqual(len(filtered), 2)
            segment = build_segments(filtered, [])[0]
            cfg = ReplayConfig(path, None, "robot.local")
            program, wait = build_segment_program(filtered, segment, 0, cfg)
            self.assertIn("def ur5e_replay_segment_000", program)
            self.assertIn("movel(p[", program)
            self.assertGreaterEqual(wait, 0.5)


if __name__ == "__main__":
    unittest.main()
