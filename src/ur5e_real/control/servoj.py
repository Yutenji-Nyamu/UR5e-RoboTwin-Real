from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..hardware.dashboard import require_external_motion_ready
from ..hardware.urscript import send_urscript

SERVOJ_LOOKAHEAD_DECLARATION = "  local servoj_lookahead_time = 0.1"
SERVOJ_GAIN_DECLARATION = "  local servoj_gain = 300"


@dataclass(frozen=True)
class ServoJStreamConfig:
    robot_host: str
    config_xml_path: Path
    robot_port: int = 30004
    frequency_hz: int = 500
    servoj_mode: int = 2
    connect_retry_s: float = 0.5
    connect_timeout_s: float = 15.0
    program_start_timeout_s: float = 2.0


def render_servoj_program(source: str, lookahead_time: float, gain: int) -> str:
    """Apply command-line servoJ tuning to the tracked robot-side program."""
    if not 0.03 <= lookahead_time <= 0.2:
        raise ValueError("servoJ lookahead_time must be in [0.03, 0.2]")
    if not 100 <= gain <= 2000:
        raise ValueError("servoJ gain must be in [100, 2000]")
    if source.count(SERVOJ_LOOKAHEAD_DECLARATION) != 1 or source.count(SERVOJ_GAIN_DECLARATION) != 1:
        raise ValueError("servoJ program tuning declarations are missing or duplicated")
    return source.replace(
        SERVOJ_LOOKAHEAD_DECLARATION,
        f"  local servoj_lookahead_time = {lookahead_time:.6f}",
    ).replace(SERVOJ_GAIN_DECLARATION, f"  local servoj_gain = {gain}")


def start_servoj_program(
    *,
    robot_host: str,
    robot_port: int,
    socket_timeout_s: float,
    program_script: Path,
    lookahead_time: float,
    gain: int,
) -> None:
    """Start the tracked robot-side loop used by every RTDE chunk executor."""
    require_external_motion_ready(robot_host)
    if not program_script.is_file():
        raise FileNotFoundError(program_script)
    program = render_servoj_program(
        program_script.read_text(encoding="utf-8"),
        lookahead_time,
        gain,
    )
    send_urscript(program, robot_host, robot_port, socket_timeout_s)
    print(
        f"[CONTROL] robot-side servoJ program started: {program_script.name}; "
        f"lookahead={lookahead_time:g}s gain={gain}"
    )


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

    def connect_and_prime(self) -> None:
        """Connect RTDE and clear the persistent watchdog before script start."""
        cfg = self.config
        if self._connection is not None:
            raise RuntimeError("RTDE controller is already connected")
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

        # Seed the registers with the measured pose before enabling servoJ, so
        # the robot-side loop never observes the temporary zero initialization.
        for index, value in enumerate(self._latest_tcp):
            setattr(setpoint, f"input_double_register_{index}", float(value))
        connection.send(setpoint)
        # RTDE input registers persist between programs.  A previous stop leaves
        # mode=3 behind, so explicitly clear it before starting the URScript.
        watchdog.input_int_register_0 = 0
        connection.send(watchdog)
        self._connection = connection
        self._setpoint = setpoint
        self._watchdog = watchdog
        print("[CONTROL] RTDE registers primed: mode=0")

    def activate(self) -> None:
        """Enable servoJ after the robot-side program has been submitted."""
        if self._connection is None or self._watchdog is None:
            raise RuntimeError("RTDE controller must be primed before activation")
        if self._running:
            raise RuntimeError("RTDE controller is already active")

        self._watchdog.input_int_register_0 = self.config.servoj_mode
        self._connection.send(self._watchdog)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ur5e-servoj", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + self.config.program_start_timeout_s
        while time.monotonic() < deadline:
            if self._latest_runtime_state == 2:
                print(
                    f"[CONTROL] RTDE servoJ running: mode={self.config.servoj_mode} "
                    f"runtime_state={self._latest_runtime_state}"
                )
                return
            time.sleep(0.01)

        runtime_state = self._latest_runtime_state
        self.stop()
        raise RuntimeError(
            "robot-side servoJ program did not enter the running state "
            f"(runtime_state={runtime_state})"
        )

    def connect_and_start(self) -> None:
        """Compatibility helper for a robot-side program that is already running."""
        self.connect_and_prime()
        self.activate()

    def _loop(self) -> None:
        while self._running and self._connection is not None:
            state = self._connection.receive()
            if state is None:
                continue
            self._latest_tcp = list(state.actual_TCP_pose)
            self._latest_runtime_state = int(state.runtime_state)
            if self._latest_runtime_state != 2:
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
        if self._running and self._latest_runtime_state != 2:
            raise RuntimeError(
                "robot-side servoJ program stopped during execution "
                f"(runtime_state={self._latest_runtime_state})"
            )
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
                # Mode 3 exits robot_programs/servoj_control_loop.script.
                self._watchdog.input_int_register_0 = 3
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
