from __future__ import annotations

import csv
import math
import threading
import time
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
    OUTPUT_FIELDS = ("timestamp", "actual_TCP_pose")

    def __init__(self, config: RtdeOutputConfig) -> None:
        self.config = config
        self.connection: Any = None

    def connect(self) -> None:
        rtde = _rtde_module()
        connection = rtde.RTDE(self.config.robot_host, self.config.robot_port)
        connection.connect()
        connection.get_controller_version()
        if not connection.send_output_setup(list(self.OUTPUT_FIELDS), frequency=self.config.frequency_hz):
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


@dataclass(frozen=True)
class RtdeRobotState:
    """Measured joint/TCP state from one packet; no IK or angle wrapping."""

    controller_time_s: float
    tcp_pose: tuple[float, ...]
    actual_q: tuple[float, ...]
    actual_qd: tuple[float, ...]
    tcp_offset: tuple[float, ...]
    host_receive_time_s: float

    def __post_init__(self) -> None:
        for name in ("controller_time_s", "host_receive_time_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("tcp_pose", "actual_q", "actual_qd", "tcp_offset"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain six finite values")
            object.__setattr__(self, name, values)


class RtdeStateClient(RtdeTcpClient):
    """Collector-only full state; legacy TCP clients keep their original recipe/API."""

    OUTPUT_FIELDS = ("timestamp", "actual_TCP_pose", "actual_q", "actual_qd", "tcp_offset")

    def receive_state(self) -> RtdeRobotState | None:
        if self.connection is None:
            raise RuntimeError("RTDE is not connected")
        packet = self.connection.receive()
        received_at = time.time()
        if packet is None:
            return None
        try:
            return RtdeRobotState(
                controller_time_s=packet.timestamp,
                tcp_pose=packet.actual_TCP_pose,
                actual_q=packet.actual_q,
                actual_qd=packet.actual_qd,
                tcp_offset=packet.tcp_offset,
                host_receive_time_s=received_at,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"invalid RTDE joint/TCP state: {exc}") from exc


class LatestRtdeTcpClient:
    """Continuously receive RTDE so policy inference never leaves stale packets queued."""

    def __init__(self, client: RtdeTcpClient) -> None:
        self.client = client
        self._latest: tuple[float, list[float]] | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.client.connect()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ur5e-rtde-state", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.stop()
            raise RuntimeError("timed out waiting for the first RTDE state")
        if self._error is not None:
            self.stop()
            raise RuntimeError("RTDE state acquisition failed") from self._error

    def _loop(self) -> None:
        while self._running:
            try:
                state = self.client.receive()
            except Exception as exc:
                if self._running:
                    self._error = exc
                    self._ready.set()
                return
            if state is None:
                continue
            latest = (state[0], list(state[1]))
            with self._lock:
                self._latest = latest
            self._ready.set()

    def read(self) -> tuple[float, list[float]] | None:
        if self._error is not None:
            raise RuntimeError("RTDE state acquisition failed") from self._error
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.client.close()
        self._thread = None


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

    def __init__(self, path: Path, *, exclusive: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x" if exclusive else "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(self.COLUMNS)
        self.handle.flush()

    def write(self, timestamp: float, pose: list[float], gripper_state: int, event_counter: int) -> None:
        self.writer.writerow([timestamp, *pose, gripper_state, event_counter])
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class RtdeStateCsvWriter(RtdeCsvWriter):
    """Append measured joints to the legacy column prefix, never overwrite raw data."""

    COLUMNS = [
        *RtdeCsvWriter.COLUMNS,
        *(f"actual_q_{index}" for index in range(6)),
        *(f"actual_qd_{index}" for index in range(6)),
        "host_receive_time_s",
    ]

    def __init__(self, path: Path) -> None:
        super().__init__(path, exclusive=True)

    def write(self, state: RtdeRobotState, gripper_state: int, event_counter: int) -> None:
        self.writer.writerow(
            [
                state.controller_time_s,
                *state.tcp_pose,
                gripper_state,
                event_counter,
                *state.actual_q,
                *state.actual_qd,
                state.host_receive_time_s,
            ]
        )
        self.handle.flush()
