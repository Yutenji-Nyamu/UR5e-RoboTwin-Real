"""Joint-only trajectory handling. TCP/rotvec math deliberately does not enter here."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def joint_vector(values: Sequence[float], *, name: str = "joints") -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be six finite joint angles in radians")
    return result


@dataclass(frozen=True)
class JointMotionConfig:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    policy_hz: float = 10.0
    servo_hz: int = 500
    max_velocity_rad_s: float = 0.6
    max_stretch: float = 2.0
    waypoint_tolerance_rad: float = 0.08
    max_lateness_s: float = 0.1

    def __post_init__(self) -> None:
        low, high = joint_vector(self.lower), joint_vector(self.upper)
        if np.any(low >= high):
            raise ValueError("joint lower bounds must be below upper bounds")
        for name in ("policy_hz", "servo_hz", "max_velocity_rad_s", "waypoint_tolerance_rad", "max_lateness_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.servo_hz != 500 or self.policy_hz != 10:
            raise ValueError("this joint driver contract is fixed to 10 Hz policy / 500 Hz servoJ")
        if not math.isfinite(self.max_stretch) or self.max_stretch < 1:
            raise ValueError("max_stretch must be finite and at least 1")

    def check(self, joints: Sequence[float]) -> np.ndarray:
        q = joint_vector(joints)
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("joint target is outside the configured demo envelope")
        return q


def plan_joint_chunk(start, targets, config: JointMotionConfig):
    """Retain endpoints; stretch slow segments instead of silently truncating motion."""
    previous = config.check(start)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("joint chunk must be a nonempty finite (N, 6) array")
    nominal_steps = round(config.servo_hz / config.policy_hz)
    segments = []
    for target in values:
        target = config.check(target)
        needed_s = float(np.max(np.abs(target - previous))) / config.max_velocity_rad_s
        steps = max(nominal_steps, math.ceil(needed_s * config.servo_hz - 1e-9))
        if steps > nominal_steps * config.max_stretch + 1e-9:
            raise ValueError("joint chunk needs excessive retiming; reject before moving")
        fractions = np.arange(1, steps + 1, dtype=np.float64)[:, None] / steps
        segments.append(previous + fractions * (target - previous))
        previous = target
    return segments


def stream_joint_chunk(
    controller,
    targets,
    config: JointMotionConfig,
    *,
    on_waypoint: Callable[[int, list[float]], None] | None = None,
    finish_after_waypoint: Callable[[int], bool] = lambda _index: False,
    cancelled: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """One clock, fresh measured feedback, and gripper callbacks only at reached knots."""
    if getattr(controller, "action_space", None) != "joint_position":
        raise ValueError("joint chunks require an explicitly joint_position controller")
    try:
        # Continue from the last command, not a lagging measured q; otherwise a
        # chunk boundary can introduce an instantaneous backwards command jump.
        segments = plan_joint_chunk(controller.get_commanded_joints(), targets, config)
        deadline = clock()
        completed = 0
        for index, segment in enumerate(segments):
            for target in segment:
                if cancelled():
                    raise InterruptedError("joint chunk cancelled")
                now = clock()
                if now - deadline > config.max_lateness_s:
                    raise TimeoutError("joint streaming clock fell behind; refusing a burst of catch-up commands")
                # Never emit a catch-up burst after a short serial/OS scheduling delay.
                deadline = max(deadline, now)
                controller.set_target_joints(target)
                deadline += 1.0 / config.servo_hz
                delay = deadline - clock()
                if delay > 0:
                    sleep(delay)
            measured = joint_vector(controller.get_latest_joints())
            if np.max(np.abs(measured - segment[-1])) > config.waypoint_tolerance_rad:
                raise RuntimeError("joint waypoint was not reached; gripper event is not applied")
            if on_waypoint is not None:
                on_waypoint(index, segment[-1].tolist())
            completed += 1
            if finish_after_waypoint(index):
                break
        segments = segments[:completed]
        return {
            "executed_waypoints": len(segments),
            "servo_steps": sum(len(segment) for segment in segments),
            "planned_duration_s": sum(len(segment) for segment in segments) / config.servo_hz,
            "retimed_waypoints": sum(len(segment) > config.servo_hz / config.policy_hz for segment in segments),
        }
    except BaseException:
        controller.stop()
        raise
