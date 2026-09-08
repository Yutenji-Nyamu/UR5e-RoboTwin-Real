import csv
import io
import itertools
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ur5e_real.collection.session import run_collection
from ur5e_real.config import CameraConfig, CollectionConfig, GripperConfig, LabConfig, RobotConfig, ServoJConfig
from ur5e_real.data.session_manifest import load_manifest
from ur5e_real.hardware.realsense import FramePair
from ur5e_real.hardware.rtde import RtdeRobotState, RtdeStateClient


def robot_state(index):
    return RtdeRobotState(
        controller_time_s=10.0 + index / 10.0,
        tcp_pose=(0.2 + index / 1000.0, 0.3, 0.4, 0.0, 3.0, 0.0),
        actual_q=(-3.5 + index / 1000.0, -1.2, 1.4, -1.7, -1.6, 3.8),
        actual_qd=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        tcp_offset=(0.0, 0.0, 0.15, 0.0, 0.0, 0.0),
        host_receive_time_s=1000.0 + index / 10.0,
    )


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cfg = LabConfig(
            robot=RobotConfig("robot.test"),
            gripper=GripperConfig(),
            cameras=CameraConfig("head-test", "wrist-test", width=8, height=8, warmup_frames=0),
            collection=CollectionConfig(self.root),
            servoj=ServoJConfig(500, Path("unused.xml"), Path("unused.script")),
            source_path=Path("test.yaml"),
        )
        module = "ur5e_real.collection.session."
        self.camera = self.stack.enter_context(patch(module + "DualColorCamera")).return_value
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.pair = FramePair(image, image)
        self.camera.read.return_value = self.pair
        rtde_factory = self.stack.enter_context(patch(module + "RtdeStateClient"))
        rtde_factory.OUTPUT_FIELDS = RtdeStateClient.OUTPUT_FIELDS
        self.rtde = rtde_factory.return_value
        self.gripper_factory = self.stack.enter_context(patch(module + "GripperSerial"))
        self.gripper = self.gripper_factory.return_value
        self.start_freedrive = self.stack.enter_context(patch(module + "start_freedrive"))
        self.stop_freedrive = self.stack.enter_context(patch(module + "stop_freedrive"))
        self.keys = MagicMock()
        self.keys.enabled = True
        terminal = self.stack.enter_context(patch(module + "TerminalKeyPoller"))
        terminal.return_value.__enter__.return_value = self.keys
        self.stack.enter_context(patch(module + "_run_id", return_value="test"))
        self.stack.enter_context(patch(module + "_code_commit", return_value="test-commit"))
        # Independent host ticks make every available image eligible for saving.
        self.stack.enter_context(patch(module + "time.monotonic", side_effect=itertools.count(0.0, 0.11)))
        self.output = self.stack.enter_context(redirect_stdout(io.StringIO()))

    @property
    def manifest_path(self):
        return self.root / "raw" / "action" / "session_test.json"

    def run_capture(self, keys):
        self.keys.poll.side_effect = keys
        self.rtde.receive_state.side_effect = [robot_state(i) for i in range(len(keys) + 1)]
        return run_collection(self.cfg, task="pick_place_cube", initial_gripper_state="open")

    def assert_resources_closed(self):
        self.rtde.close.assert_called_once_with()
        self.camera.stop.assert_called_once_with()

    def test_complete_capture_persists_joint_tcp_events_metadata_and_release_tail(self):
        manifest = load_manifest(self.run_capture(["c", "c", "o", *([None] * 10), "q"]))
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["recording_status"], "completed")
        self.assertEqual(manifest["outcome"], "unreviewed")
        self.assertEqual(manifest["code_commit"], "test-commit")
        self.assertEqual(manifest["state_recording"]["joint_source"], "actual_q")
        self.assertEqual(manifest["state_recording"]["sample_relation"], "same_rtde_packet")
        self.assertEqual(manifest["initial_robot_state"]["actual_q"], list(robot_state(0).actual_q))
        self.assertEqual(manifest["initial_robot_state"]["tcp_offset"], list(robot_state(0).tcp_offset))
        self.assertEqual(manifest["counts"], {"rtde_samples": 13, "gripper_events": 3, "frame_pairs": 13})
        with Path(manifest["paths"]["rtde"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for index, row in enumerate(rows, 1):
            state = robot_state(index)
            self.assertEqual(float(row["controller_time_s"]), state.controller_time_s)
            self.assertEqual(float(row["actual_q_0"]), state.actual_q[0])
            self.assertEqual(float(row["tcp_x"]), state.tcp_pose[0])
            self.assertEqual(float(row["actual_qd_0"]), state.actual_qd[0])
            self.assertEqual(float(row["host_receive_time_s"]), state.host_receive_time_s)
        self.assertEqual([int(row["gripper_event_counter"]) for row in rows[:3]], [1, 2, 3])
        with Path(manifest["paths"]["gripper_events"]).open(newline="", encoding="utf-8") as handle:
            events = list(csv.DictReader(handle))
        self.assertEqual([row["event"] for row in events], ["close", "close", "open"])
        with Path(manifest["paths"]["sync"]).open(newline="", encoding="utf-8") as handle:
            sync = list(csv.DictReader(handle))
        self.assertEqual([row["controller_time_s"] for row in sync], [row["controller_time_s"] for row in rows])
        self.assertEqual(len(list(Path(manifest["paths"]["head_frames"]).glob("*.png"))), 13)
        self.assertEqual(manifest["quality"]["last_open_to_last_frame_s"], 1.0)
        self.assertTrue(manifest["quality"]["release_tail_complete"])
        self.assertIn("[STATE] schema=3", self.output.getvalue())
        self.assertNotIn("[WARN]", self.output.getvalue())
        self.assertEqual(self.gripper.close.call_count, 2)
        self.gripper.open.assert_called_once_with()
        self.start_freedrive.assert_called_once()
        self.stop_freedrive.assert_called_once()
        self.assert_resources_closed()

    def test_camera_gap_does_not_drop_robot_state_or_claim_release_images(self):
        self.camera.read.side_effect = [self.pair, None]
        manifest = load_manifest(self.run_capture(["c", "o", "q"]))
        self.assertEqual(manifest["counts"], {"rtde_samples": 2, "gripper_events": 2, "frame_pairs": 1})
        self.assertFalse(manifest["quality"]["release_tail_complete"])
        self.assertLess(manifest["quality"]["last_open_to_last_frame_s"], 0.0)
        self.assertIn("less than 1 second of images", self.output.getvalue())

    def test_short_tail_warns_but_never_delays_quit(self):
        manifest = load_manifest(self.run_capture(["o", None, "q"]))
        self.assertEqual(manifest["counts"]["frame_pairs"], 2)
        self.assertEqual(manifest["quality"]["last_open_to_last_frame_s"], 0.1)
        self.assertFalse(manifest["quality"]["release_tail_complete"])
        self.assertEqual(manifest["recording_status"], "completed")
        self.assertEqual(manifest["outcome"], "unreviewed")
        self.assertIn("less than 1 second of images", self.output.getvalue())

    def test_failed_image_write_does_not_count_an_unsaved_pair(self):
        with patch("cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "failed to write head"):
                self.run_capture(["o", "q"])
        manifest = load_manifest(self.manifest_path)
        self.assertEqual(manifest["counts"]["rtde_samples"], 1)
        self.assertEqual(manifest["counts"]["frame_pairs"], 0)
        self.assertEqual(manifest["recording_status"], "failed")
        self.assertFalse(manifest["quality"]["release_tail_complete"])
        self.assert_resources_closed()

    def test_invalid_initial_state_stops_before_freedrive_or_gripper(self):
        self.rtde.receive_state.side_effect = RuntimeError("invalid RTDE joint/TCP state: missing actual_q")
        with self.assertRaisesRegex(RuntimeError, "missing actual_q"):
            run_collection(self.cfg, task="pick_place_cube", initial_gripper_state="open")
        manifest = load_manifest(self.manifest_path)
        self.assertEqual(manifest["recording_status"], "failed")
        self.assertEqual(manifest["counts"]["rtde_samples"], 0)
        self.assertIsNone(manifest["initial_robot_state"])
        self.start_freedrive.assert_not_called()
        self.gripper_factory.assert_not_called()
        self.assert_resources_closed()

    def test_invalid_midstream_state_preserves_prior_rows_and_finalizes_failure(self):
        self.rtde.receive_state.side_effect = [
            robot_state(0),
            robot_state(1),
            RuntimeError("invalid RTDE joint/TCP state"),
        ]
        self.keys.poll.return_value = "c"
        with self.assertRaisesRegex(RuntimeError, "invalid RTDE"):
            run_collection(self.cfg, task="pick_place_cube", initial_gripper_state="open")
        manifest = load_manifest(self.manifest_path)
        self.assertEqual(manifest["recording_status"], "failed")
        self.assertEqual(manifest["counts"]["rtde_samples"], 1)
        self.assertEqual(manifest["counts"]["frame_pairs"], 1)
        self.stop_freedrive.assert_called_once()
        self.gripper.shutdown.assert_called_once()
        self.assert_resources_closed()

    def test_nonmonotonic_controller_time_is_rejected(self):
        self.rtde.receive_state.side_effect = [robot_state(0), robot_state(0)]
        with self.assertRaisesRegex(RuntimeError, "controller time did not advance"):
            run_collection(self.cfg, task="pick_place_cube", initial_gripper_state="open")
        self.assertEqual(load_manifest(self.manifest_path)["recording_status"], "failed")
        self.assert_resources_closed()

    def test_keyboard_interrupt_finalizes_without_release_claim(self):
        self.rtde.receive_state.side_effect = [robot_state(0), KeyboardInterrupt()]
        path = run_collection(self.cfg, task="pick_place_cube", initial_gripper_state="open")
        manifest = load_manifest(path)
        self.assertEqual(manifest["recording_status"], "interrupted")
        self.assertIsNone(manifest["quality"]["release_tail_complete"])
        self.stop_freedrive.assert_called_once()
        self.assert_resources_closed()

    def test_run_id_collision_never_overwrites_raw_files(self):
        path = self.manifest_path.parent / "rtde_tcp_gripper_test.csv"
        path.parent.mkdir(parents=True)
        path.write_text("original recording\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            run_collection(self.cfg, task="pick_place_cube", initial_gripper_state="open")
        self.assertEqual(path.read_text(encoding="utf-8"), "original recording\n")
        self.camera.start.assert_not_called()
        self.rtde.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
