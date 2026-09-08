"""Explicit joint servoJ driver with fresh program identity and fail-closed telemetry."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..hardware.dashboard import require_external_motion_ready
from ..hardware.rtde import RtdeRobotState
from ..hardware.urscript import send_urscript
from .joint import JointMotionConfig, joint_vector
from .servoj import _imports, render_servoj_program

PROGRAM_MARKER = "# UR5E_ACTION_SPACE: joint_position"
PROGRAM_ID = 10505


@dataclass(frozen=True)
class JointServoJConfig:
    robot_host: str
    config_xml: Path
    program_script: Path
    motion: JointMotionConfig
    tcp_lower: tuple[float, ...]
    tcp_upper: tuple[float, ...]
    tcp_offset: tuple[float, ...]
    robot_port: int = 30004
    script_port: int = 30001
    socket_timeout_s: float = 10.0
    startup_timeout_s: float = 3.0
    state_timeout_s: float = 0.5
    command_timeout_s: float = 2.0
    max_tracking_error_rad: float = 0.15
    lookahead_time: float = 0.1
    gain: int = 300

    def __post_init__(self):
        for name in ("startup_timeout_s", "state_timeout_s", "command_timeout_s", "max_tracking_error_rad"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        low, high = np.asarray(self.tcp_lower), np.asarray(self.tcp_upper)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
            raise ValueError("TCP monitoring bounds must contain finite lower/upper XYZ")
        joint_vector(self.tcp_offset, name="TCP offset")


def validate_program_ack(packet, nonce: int) -> bool:
    return (
        int(packet.runtime_state) == 2
        and int(packet.output_int_register_0) == PROGRAM_ID
        and int(packet.output_int_register_1) == 2
        and int(packet.output_int_register_2) == nonce
    )


class JointServoJController:
    action_space = "joint_position"

    def __init__(self, config: JointServoJConfig):
        self.config = config
        self._connection = None
        self._setpoint = None
        self._watchdog = None
        self._state: RtdeRobotState | None = None
        self._target = None
        self._state_at = 0.0
        self._command_at = 0.0
        self._nonce = secrets.randbelow(2_000_000_000) + 1
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._primed = False

    def _accept_state(self, packet) -> None:
        state = RtdeRobotState(
            packet.timestamp,
            packet.actual_TCP_pose,
            packet.actual_q,
            packet.actual_qd,
            packet.tcp_offset,
            time.time(),
        )
        self.config.motion.check(state.actual_q)
        xyz = np.asarray(state.tcp_pose[:3])
        if np.any(xyz < self.config.tcp_lower) or np.any(xyz > self.config.tcp_upper):
            raise RuntimeError("measured TCP left the configured demo workspace")
        if not np.allclose(state.tcp_offset, self.config.tcp_offset, atol=1e-5, rtol=0):
            raise RuntimeError("controller TCP offset differs from the demonstration contract")
        with self._lock:
            if self._state is not None and state.controller_time_s <= self._state.controller_time_s:
                raise RuntimeError("joint controller telemetry time stopped or moved backwards")
            self._state = state
            self._state_at = time.monotonic()

    def connect_and_prime(self) -> None:
        if self._connection is not None or self._stop_event.is_set():
            raise RuntimeError("use a fresh joint controller instance for each execution")
        # Must precede every register write: a still-running TCP script would interpret q as a pose.
        require_external_motion_ready(self.config.robot_host)
        rtde, rtde_config = _imports()
        recipes = rtde_config.ConfigFile(str(self.config.config_xml))
        connection = rtde.RTDE(self.config.robot_host, self.config.robot_port)
        self._connection = connection
        try:
            connection.connect()
            connection.get_controller_version()
            names, types = recipes.get_recipe("state")
            if not connection.send_output_setup(names, types, self.config.motion.servo_hz):
                raise RuntimeError("joint RTDE output recipe rejected")
            self._setpoint = connection.send_input_setup(*recipes.get_recipe("setp"))
            self._watchdog = connection.send_input_setup(*recipes.get_recipe("watchdog"))
            if self._setpoint is None or self._watchdog is None:
                raise RuntimeError("joint RTDE input recipe unavailable; another controller may own it")
            if not connection.send_start():
                raise RuntimeError("joint RTDE send_start failed")
            packet = connection.receive()
            if packet is None:
                raise RuntimeError("no initial joint RTDE packet")
            if int(packet.runtime_state) != 1:
                raise RuntimeError("RTDE runtime is not STOPPED; refusing to prime joint registers")
            self._accept_state(packet)
            self._target = joint_vector(self._state.actual_q)
            for index, value in enumerate(self._target):
                setattr(self._setpoint, f"input_double_register_{index}", float(value))
            self._watchdog.input_int_register_0 = 0
            self._watchdog.input_int_register_1 = 0
            self._watchdog.input_int_register_2 = self._nonce
            if not connection.send(self._setpoint) or not connection.send(self._watchdog):
                raise RuntimeError("failed to prime joint RTDE registers")
            self._primed = True
            print("[CONTROL] joint registers primed with measured q: mode=0")
        except BaseException:
            self.stop()
            raise

    def start(self) -> None:
        source = self.config.program_script.read_text(encoding="utf-8")
        if source.count(PROGRAM_MARKER) != 1 or "get_inverse_kin(" in source:
            raise ValueError("refusing a non-joint robot-side script")
        program = render_servoj_program(source, self.config.lookahead_time, self.config.gain)
        require_external_motion_ready(self.config.robot_host)
        try:
            if not self._primed:
                self.connect_and_prime()
            self._check_health()  # Never start from an old priming sample/setpoint.
            send_urscript(program, self.config.robot_host, self.config.script_port, self.config.socket_timeout_s)
            self._watchdog.input_int_register_0 = 2
            self._command_at = time.monotonic()
            self._connection.send(self._watchdog)
            self._thread = threading.Thread(target=self._loop, name="ur5e-joint-servoj", daemon=True)
            self._thread.start()
            if not self._ready.wait(self.config.startup_timeout_s):
                raise TimeoutError("joint script did not acknowledge the fresh nonce and mode=2")
            self._check_health()
            print(f"[CONTROL] joint servoJ running: mode=2 runtime=2 program={PROGRAM_ID}")
        except BaseException:
            self.stop()
            raise

    def _loop(self):
        activated = False
        try:
            while not self._stop_event.is_set():
                packet = self._connection.receive()
                if packet is None:
                    raise RuntimeError("joint RTDE stream closed")
                self._accept_state(packet)
                acknowledged = validate_program_ack(packet, self._nonce)
                if activated and not acknowledged:
                    raise RuntimeError("joint program stopped or identity changed")
                activated = activated or acknowledged
                if time.monotonic() - self._command_at > self.config.command_timeout_s:
                    raise TimeoutError("joint command producer timed out")
                with self._lock:
                    target = self._target.copy()
                    measured = np.asarray(self._state.actual_q)
                if np.max(np.abs(target - measured)) > self.config.max_tracking_error_rad:
                    raise RuntimeError("joint tracking error exceeded the configured limit")
                for index, value in enumerate(target):
                    setattr(self._setpoint, f"input_double_register_{index}", float(value))
                self._watchdog.input_int_register_1 = (self._watchdog.input_int_register_1 + 1) % 2_000_000_000
                if not self._connection.send(self._setpoint) or not self._connection.send(self._watchdog):
                    raise RuntimeError("failed to send joint setpoint/heartbeat")
                if acknowledged:
                    self._ready.set()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._stop_event.set()
            self._send_stop()

    def _check_health(self):
        if self._error is not None:
            raise RuntimeError(f"joint controller failed: {self._error}") from self._error
        if self._stop_event.is_set() or self._state is None:
            raise RuntimeError("joint controller is stopped or has no state")
        if time.monotonic() - self._state_at > self.config.state_timeout_s:
            raise TimeoutError("joint controller state is stale")

    def get_latest_joints(self):
        self._check_health()
        with self._lock:
            return list(self._state.actual_q)

    def get_latest_state(self):
        self._check_health()
        with self._lock:
            return self._state

    def get_commanded_joints(self):
        self._check_health()
        with self._lock:
            return self._target.copy()

    def set_target_joints(self, joints):
        self._check_health()
        if not self._ready.is_set() or self._thread is None:
            raise RuntimeError("joint controller has not activated")
        target = self.config.motion.check(joints)
        with self._lock:
            self._target = target.copy()
            self._command_at = time.monotonic()

    def _send_stop(self):
        if self._connection is not None and self._watchdog is not None and self._primed:
            try:
                self._watchdog.input_int_register_0 = 3
                self._connection.send(self._watchdog)
            except Exception:
                pass  # Robot-side heartbeat watchdog also stops on a lost link.

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.25)
        if self._thread is None or not self._thread.is_alive():
            self._send_stop()
        if self._connection is not None:
            try:
                self._connection.disconnect()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self._connection = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
