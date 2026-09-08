import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ur5e_real.data.convert_hdf5 import load_actions
from ur5e_real.data.schema import ACTION_COLUMNS, RAW_SCHEMA_VERSION, SCHEMA_VERSION
from ur5e_real.hardware.rtde import (
    RtdeCsvWriter,
    RtdeOutputConfig,
    RtdeRobotState,
    RtdeStateClient,
    RtdeStateCsvWriter,
    RtdeTcpClient,
)
from ur5e_real.replay import load_action_rows


def packet(timestamp=12.5):
    return SimpleNamespace(
        timestamp=timestamp,
        actual_TCP_pose=[0.2, 0.3, 0.4, -0.1, 0.2, -0.3],
        actual_q=[-3.5, -1.2, 1.4, -1.7, -1.6, 3.8],
        actual_qd=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        tcp_offset=[0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
    )


class RtdeStateTest(unittest.TestCase):
    def test_legacy_client_recipe_and_tuple_api_unchanged(self):
        connection = MagicMock()
        original = packet()
        connection.receive.return_value = original
        with patch("ur5e_real.hardware.rtde._rtde_module") as module:
            module.return_value.RTDE.return_value = connection
            with RtdeTcpClient(RtdeOutputConfig("robot.test")) as client:
                connection.send_output_setup.assert_called_once_with(["timestamp", "actual_TCP_pose"], frequency=10.0)
                self.assertEqual(client.receive(), (12.5, original.actual_TCP_pose))

    def test_full_state_is_from_one_packet_without_joint_wrapping(self):
        connection = MagicMock()
        original = packet()
        connection.receive.return_value = original
        with patch("ur5e_real.hardware.rtde._rtde_module") as module:
            module.return_value.RTDE.return_value = connection
            with RtdeStateClient(RtdeOutputConfig("robot.test")) as client:
                connection.send_output_setup.assert_called_once_with(
                    ["timestamp", "actual_TCP_pose", "actual_q", "actual_qd", "tcp_offset"], frequency=10.0
                )
                with patch("ur5e_real.hardware.rtde.time.time", return_value=12345.0):
                    state = client.receive_state()
                connection.receive.assert_called_once_with()
                self.assertEqual(state.controller_time_s, 12.5)
                self.assertEqual(state.host_receive_time_s, 12345.0)
                self.assertEqual(state.actual_q, tuple(original.actual_q))
                self.assertEqual(state.actual_qd, tuple(original.actual_qd))
                self.assertEqual(state.tcp_pose, tuple(original.actual_TCP_pose))
                self.assertEqual(state.tcp_offset, tuple(original.tcp_offset))
                original.actual_q[0] = 99.0
                self.assertEqual(state.actual_q[0], -3.5)
        connection.disconnect.assert_called_once_with()

    def test_missing_or_invalid_state_never_becomes_zero_joints(self):
        for field, value in (
            ("actual_q", None),
            ("actual_q", [0.0] * 5),
            ("actual_q", [float("nan")] * 6),
            ("actual_qd", [float("inf")] * 6),
            ("actual_TCP_pose", [0.0] * 7),
            ("tcp_offset", None),
            ("timestamp", float("nan")),
        ):
            with self.subTest(field=field, value=value):
                original = packet()
                if value is None:
                    delattr(original, field)
                else:
                    setattr(original, field, value)
                client = RtdeStateClient(RtdeOutputConfig("robot.test"))
                client.connection = MagicMock()
                client.connection.receive.return_value = original
                with self.assertRaisesRegex(RuntimeError, "invalid RTDE joint/TCP state"):
                    client.receive_state()

    def test_not_connected_and_closed_connection(self):
        client = RtdeStateClient(RtdeOutputConfig("robot.test"))
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            client.receive_state()
        client.connection = MagicMock()
        client.connection.receive.return_value = None
        self.assertIsNone(client.receive_state())

    def test_recipe_rejection_disconnects(self):
        connection = MagicMock()
        connection.send_output_setup.return_value = False
        with patch("ur5e_real.hardware.rtde._rtde_module") as module:
            module.return_value.RTDE.return_value = connection
            client = RtdeStateClient(RtdeOutputConfig("robot.test"))
            with self.assertRaisesRegex(RuntimeError, "configure RTDE"):
                client.connect()
        connection.disconnect.assert_called_once_with()
        self.assertIsNone(client.connection)

    def test_v3_csv_round_trip_and_legacy_tcp_consumers(self):
        original = packet()
        state = RtdeRobotState(
            original.timestamp,
            original.actual_TCP_pose,
            original.actual_q,
            original.actual_qd,
            original.tcp_offset,
            12345.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtde_tcp_gripper_test.csv"
            writer = RtdeStateCsvWriter(path)
            writer.write(state, 1, 2)
            writer.close()
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames[: len(ACTION_COLUMNS)], ACTION_COLUMNS)
                row = next(reader)
            self.assertNotIn(None, row)
            self.assertEqual([float(row[f"actual_q_{i}"]) for i in range(6)], original.actual_q)
            self.assertEqual([float(row[f"actual_qd_{i}"]) for i in range(6)], original.actual_qd)
            self.assertEqual(float(row["host_receive_time_s"]), 12345.0)
            self.assertEqual(int(row["gripper_event_counter"]), 2)
            times, poses, gripper = load_actions(path)
            np.testing.assert_allclose(times, [original.timestamp])
            np.testing.assert_allclose(poses, [original.actual_TCP_pose])
            np.testing.assert_allclose(gripper, [1])
            self.assertEqual(load_action_rows(path)[0]["pose"], original.actual_TCP_pose)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                RtdeStateCsvWriter(path)
            self.assertEqual(path.read_bytes(), before)

    def test_legacy_csv_writer_and_converted_schema_unchanged(self):
        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertEqual(RAW_SCHEMA_VERSION, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            writer = RtdeCsvWriter(path)
            writer.write(1.0, [0.0] * 6, 0, 0)
            writer.close()
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ACTION_COLUMNS)
                self.assertNotIn("actual_q_0", next(reader))


if __name__ == "__main__":
    unittest.main()
