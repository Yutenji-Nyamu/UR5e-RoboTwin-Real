from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RtdeOutputConfig:
    robot_host: str
    robot_port: int = 30004
    frequency_hz: float = 10.0


def _rtde_module() -> Any:
    try:
        import rtde.rtde as rtde
    except ImportError as exc:
        raise RuntimeError("UrRtde is required for RTDE communication") from exc
    return rtde


class RtdeTcpClient:
    def __init__(self, config: RtdeOutputConfig) -> None:
        self.config = config
        self.connection: Any = None

    def connect(self) -> None:
        rtde = _rtde_module()
        connection = rtde.RTDE(self.config.robot_host, self.config.robot_port)
        connection.connect()
        connection.get_controller_version()
        if not connection.send_output_setup(
            ["timestamp", "actual_TCP_pose"], frequency=self.config.frequency_hz
        ):
            connection.disconnect()
            raise RuntimeError("failed to configure RTDE output")
        if not connection.send_start():
            connection.disconnect()
            raise RuntimeError("failed to start RTDE synchronization")
        self.connection = connection

    def receive(self) -> tuple[float, list[float]] | None:
        if self.connection is None:
            raise RuntimeError("RTDE is not connected")
        state = self.connection.receive()
        if state is None:
            return None
        return float(state.timestamp), list(state.actual_TCP_pose)

    def close(self) -> None:
        if self.connection is None:
            return
        try:
            self.connection.send_pause()
        except Exception:
            pass
        try:
            self.connection.disconnect()
        finally:
            self.connection = None

    def __enter__(self) -> "RtdeTcpClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class RtdeCsvWriter:
    COLUMNS = [
        "controller_time_s",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_rx",
        "tcp_ry",
        "tcp_rz",
        "gripper_state",
        "gripper_event_counter",
    ]

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(self.COLUMNS)
        self.handle.flush()

    def write(self, timestamp: float, pose: list[float], gripper_state: int, event_counter: int) -> None:
        self.writer.writerow([timestamp, *pose, gripper_state, event_counter])
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
