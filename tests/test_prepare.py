import unittest
from unittest.mock import MagicMock, patch

from ur5e_real.control.prepare import pose_error, run_prepare


class PrepareTest(unittest.TestCase):
    def test_pose_error_separates_translation_and_rotation(self):
        translation, rotation = pose_error([0, 0, 0, 0, 0, 0], [0.003, 0.004, 0, 0, 0, 0.1])
        self.assertAlmostEqual(translation, 0.005)
        self.assertAlmostEqual(rotation, 0.1)

    def test_equivalent_rotation_vector_branch_has_zero_error(self):
        _, rotation = pose_error([0, 0, 0, 0, 0, 3.0], [0, 0, 0, 0, 0, 3.0 - 2.0 * 3.141592653589793])
        self.assertAlmostEqual(rotation, 0.0)

    def test_failed_handshake_never_sends_home_motion_or_opens_gripper(self):
        cfg = MagicMock()
        cfg.robot.home_tcp_pose = [0.0] * 6
        with (
            patch("ur5e_real.control.prepare.require_external_motion_ready"),
            patch("ur5e_real.control.prepare.RtdeTcpClient") as client,
            patch("ur5e_real.control.prepare.move_linear") as move,
            patch("ur5e_real.control.prepare.GripperSerial") as gripper,
        ):
            client.return_value.connect.side_effect = RuntimeError("RTDE startup failed")
            with self.assertRaisesRegex(RuntimeError, "RTDE startup failed"):
                run_prepare(cfg, execute=True)
        move.assert_not_called()
        gripper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
