from __future__ import annotations

import math
import time
from collections.abc import Sequence

from ..config import LabConfig
from ..hardware.dashboard import require_external_motion_ready
from ..hardware.gripper import GripperSerial
from ..hardware.rtde import RtdeOutputConfig, RtdeTcpClient
from ..hardware.urscript import move_linear

HOME_ACCELERATION = 0.3
HOME_VELOCITY = 0.04
HOME_TIMEOUT_S = 30.0
TRANSLATION_TOLERANCE_M = 0.005
ROTATION_TOLERANCE_RAD = 0.05


def pose_error(current: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    translation = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(current[:3], target[:3])))

    def quaternion(rotation_vector: Sequence[float]) -> tuple[float, float, float, float]:
        angle = math.sqrt(sum(float(value) ** 2 for value in rotation_vector))
        if angle < 1e-12:
            return 1.0, 0.0, 0.0, 0.0
        scale = math.sin(angle / 2.0) / angle
        return (
            math.cos(angle / 2.0),
            *(float(value) * scale for value in rotation_vector),
        )

    current_q = quaternion(current[3:])
    target_q = quaternion(target[3:])
    dot = min(1.0, abs(sum(a * b for a, b in zip(current_q, target_q))))
    rotation = 2.0 * math.acos(dot)
    return translation, rotation


def _wait_for_home(client: RtdeTcpClient, target: Sequence[float]) -> tuple[list[float], float, float]:
    started = time.monotonic()
    stable_samples = 0
    latest = list(target)
    translation = math.inf
    rotation = math.inf
    while time.monotonic() - started < HOME_TIMEOUT_S:
        sample = client.receive()
        if sample is None:
            raise RuntimeError("RTDE connection closed while moving home")
        _, latest = sample
        translation, rotation = pose_error(latest, target)
        if translation <= TRANSLATION_TOLERANCE_M and rotation <= ROTATION_TOLERANCE_RAD:
            stable_samples += 1
            if stable_samples >= 3:
                return latest, translation, rotation
        else:
            stable_samples = 0
    raise TimeoutError(
        f"home target not reached in {HOME_TIMEOUT_S:.0f}s; "
        f"translation_error={translation:.4f}m rotation_error={rotation:.4f}rad"
    )


def run_prepare(cfg: LabConfig, *, execute: bool = False) -> None:
    target = cfg.robot.home_tcp_pose
    if target is None:
        raise ValueError("robot.home_tcp_pose is not configured")
    print(f"home TCP: {[round(value, 6) for value in target]}")
    print(f"moveL: a={HOME_ACCELERATION:.2f}m/s^2 v={HOME_VELOCITY:.2f}m/s")
    print("after arrival: open gripper")
    if not execute:
        print("[DRY RUN] add --execute only after checking the workcell and PolyScope Remote Control")
        return

    require_external_motion_ready(cfg.robot.host)
    client = RtdeTcpClient(
        RtdeOutputConfig(cfg.robot.host, cfg.robot.rtde_port, cfg.robot.rtde_frequency_hz)
    )
    client.connect()
    try:
        sample = client.receive()
        if sample is None:
            raise RuntimeError("RTDE returned no current TCP pose")
        _, current = sample
        translation, rotation = pose_error(current, target)
        print(f"current error: translation={translation:.4f}m rotation={rotation:.4f}rad")
        if translation > TRANSLATION_TOLERANCE_M or rotation > ROTATION_TOLERANCE_RAD:
            move_linear(
                cfg.robot.host,
                target,
                acceleration=HOME_ACCELERATION,
                velocity=HOME_VELOCITY,
                port=cfg.robot.script_port,
                timeout_s=cfg.robot.socket_timeout_s,
            )
        _, translation, rotation = _wait_for_home(client, target)
        print(f"[HOME] reached: translation={translation:.4f}m rotation={rotation:.4f}rad")
    finally:
        client.close()

    with GripperSerial(cfg.gripper.port, cfg.gripper.baudrate, cfg.gripper.timeout_s) as gripper:
        gripper.open()
        time.sleep(0.8)
    print("[READY] home reached and gripper opened")
