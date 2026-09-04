import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr

from ur5e_real.adapters.robotwin_act.process_data import process_run
from ur5e_real.adapters.robotwin_dp.process_data import process_run as process_dp_run
from ur5e_real.data.convert_hdf5 import align_nearest, convert_raw_sessions


class AlignmentTest(unittest.TestCase):
    def test_nearest_action_indices(self):
        actions = np.asarray([0.0, 1.0, 2.0, 3.0])
        queries = np.asarray([-1.0, 0.49, 0.51, 2.5, 5.0])
        np.testing.assert_array_equal(align_nearest(actions, queries), [0, 0, 1, 2, 3])

    def test_empty_actions_rejected(self):
        with self.assertRaises(ValueError):
            align_nearest(np.asarray([]), np.asarray([0.0]))

    def test_raw_to_robotwin_act_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action_dir = root / "raw" / "action"
            camera_dir = root / "raw" / "camera" / "cam_dual_test"
            head_dir = camera_dir / "head"
            wrist_dir = camera_dir / "wrist"
            for path in (action_dir, head_dir, wrist_dir):
                path.mkdir(parents=True, exist_ok=True)

            with (action_dir / "rtde_tcp_gripper_test.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "controller_time_s",
                        "tcp_x",
                        "tcp_y",
                        "tcp_z",
                        "tcp_rx",
                        "tcp_ry",
                        "tcp_rz",
                        "gripper_state",
                    ]
                )
                writer.writerow([0.0, 0, 0, 0, 0, 0, 0, 0])
                writer.writerow([1.0, 1, 2, 3, 0.4, 0.5, 0.6, 1])
                writer.writerow([2.0, 2, 3, 4, 0.5, 0.6, 0.7, 0])

            with (action_dir / "sync_action_cam_test.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["controller_time_s", "frame_idx", "head_image", "wrist_image"])
                for index, timestamp in enumerate((0.0, 1.0, 2.0), start=1):
                    name = f"frame_{index:05d}.png"
                    image = np.full((8, 8, 3), index * 30, dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(head_dir / name), image))
                    self.assertTrue(cv2.imwrite(str(wrist_dir / name), image))
                    writer.writerow([timestamp, index, name, name])

            (action_dir / "session_test.json").write_text(
                json.dumps({"run_id": "test", "task": "task", "outcome": "success"}),
                encoding="utf-8",
            )

            converted = convert_raw_sessions(
                action_dir,
                root / "raw" / "camera",
                root / "converted",
                "task",
                "simple",
            )
            episode = converted / "task" / "simple" / "data" / "episode0.hdf5"
            with h5py.File(episode, "r") as handle:
                self.assertEqual(handle["joint_action/right_arm"].shape, (3, 6))
                self.assertEqual(handle["joint_action/vector"].shape, (3, 14))
                self.assertEqual(handle["observation/head_camera/rgb"].shape, (3,))

            act_root = root / "RoboTwin" / "policy" / "ACT"
            act_root.mkdir(parents=True)
            output = process_run(converted, root / "RoboTwin", "task", "simple", 1)
            with h5py.File(output / "episode_0.hdf5", "r") as handle:
                self.assertEqual(handle["observations/qpos"].shape, (2, 14))
                self.assertEqual(handle["action"].shape, (2, 14))
                np.testing.assert_allclose(handle["observations/qpos"][0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                np.testing.assert_allclose(
                    handle["action"][0],
                    [1, 2, 3, 0.4, 0.5, 0.6, 0, 1, 2, 3, 0.4, 0.5, 0.6, 1],
                )

            dp_output = process_dp_run(converted, "task", "simple", 1)
            dp = zarr.open_group(str(dp_output), mode="r")
            self.assertEqual(dp["data/head_camera"].shape, (2, 3, 240, 320))
            self.assertEqual(dp["data/state"].shape, (2, 14))
            self.assertEqual(dp["data/action"].shape, (2, 14))
            np.testing.assert_array_equal(dp["meta/episode_ends"][:], [2])
            np.testing.assert_allclose(dp["data/action"][0], dp["data/state"][1])


if __name__ == "__main__":
    unittest.main()
