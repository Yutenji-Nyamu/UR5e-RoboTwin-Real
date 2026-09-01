from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ServoJStreamConfig:
    robot_host: str
    config_xml_path: Path
    robot_port: int = 30004
    frequency_hz: int = 500
    servoj_mode: int = 2
    connect_retry_s: float = 0.5
    connect_timeout_s: float = 15.0


def _imports() -> tuple[Any, Any]:
    try:
        import rtde.rtde as rtde
        import rtde.rtde_config as rtde_config
    except ImportError as exc:
        raise RuntimeError("UrRtde is required for servoJ streaming") from exc
    return rtde, rtde_config


class ServoJController:
    """Background RTDE setpoint streamer for the robot-side servoJ loop."""

    def __init__(self, config: ServoJStreamConfig) -> None:
        self.config = config
        self._connection: Any = None
        self._setpoint: Any = None
        self._watchdog: Any = None
        self._latest_tcp: list[float] | None = None
        self._latest_runtime_state: int | None = None
        self._target_tcp: list[float] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def connect_and_start(self) -> None:
        cfg = self.config
        if not cfg.config_xml_path.is_file():
            raise FileNotFoundError(cfg.config_xml_path)
        rtde, rtde_config = _imports()
        recipes = rtde_config.ConfigFile(str(cfg.config_xml_path))
        state_names, state_types = recipes.get_recipe("state")
        setpoint_names, setpoint_types = recipes.get_recipe("setp")
        watchdog_names, watchdog_types = recipes.get_recipe("watchdog")
        connection = rtde.RTDE(cfg.robot_host, cfg.robot_port)

        started = time.monotonic()
        while True:
            try:
                connection.connect()
                connection.get_controller_version()
                break
            except Exception as exc:
                if time.monotonic() - started >= cfg.connect_timeout_s:
                    raise RuntimeError(
                        f"RTDE connection timed out for {cfg.robot_host}:{cfg.robot_port}"
                    ) from exc
                time.sleep(cfg.connect_retry_s)

        connection.send_output_setup(state_names, state_types, cfg.frequency_hz)
        setpoint = connection.send_input_setup(setpoint_names, setpoint_types)
        watchdog = connection.send_input_setup(watchdog_names, watchdog_types)
        for index in range(6):
            setattr(setpoint, f"input_double_register_{index}", 0.0)
        if hasattr(setpoint, "input_bit_registers0_to_31"):
            setpoint.input_bit_registers0_to_31 = 0
        watchdog.input_int_register_0 = 0
        if not connection.send_start():
            connection.disconnect()
            raise RuntimeError("RTDE send_start() failed")

        state = connection.receive()
        if state is None:
            connection.disconnect()
            raise RuntimeError("RTDE returned no state after startup")
        self._latest_tcp = list(state.actual_TCP_pose)
        self._latest_runtime_state = int(state.runtime_state)
        self._target_tcp = list(self._latest_tcp)

        watchdog.input_int_register_0 = cfg.servoj_mode
        connection.send(watchdog)
        self._connection = connection
        self._setpoint = setpoint
        self._watchdog = watchdog
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ur5e-servoj", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running and self._connection is not None:
            state = self._connection.receive()
            if state is None:
                continue
            self._latest_tcp = list(state.actual_TCP_pose)
            self._latest_runtime_state = int(state.runtime_state)
            if self._latest_runtime_state <= 1:
                continue
            with self._lock:
                target = None if self._target_tcp is None else list(self._target_tcp)
            if target is not None:
                for index, value in enumerate(target):
                    setattr(self._setpoint, f"input_double_register_{index}", float(value))
                self._connection.send(self._setpoint)

    def set_target_tcp(self, pose: Sequence[float]) -> None:
        if len(pose) != 6:
            raise ValueError("pose must contain six values")
        with self._lock:
            self._target_tcp = [float(value) for value in pose]

    def get_latest_tcp(self) -> list[float] | None:
        return None if self._latest_tcp is None else list(self._latest_tcp)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._watchdog is not None and self._connection is not None:
            try:
                self._watchdog.input_int_register_0 = 0
                self._connection.send(self._watchdog)
            except Exception:
                pass
        if self._connection is not None:
            try:
                self._connection.send_pause()
            except Exception:
                pass
            try:
                self._connection.disconnect()
            except Exception:
                pass
        self._connection = None
        self._thread = None

    def __enter__(self) -> "ServoJController":
        self.connect_and_start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
