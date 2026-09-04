from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..data.schema import nearest_rotation_vector
from .trajectory import interpolate_pose


@dataclass(frozen=True)
class ChunkStreamConfig:
    policy_hz: float = 10.0
    servo_hz: int = 500
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.5
    action_delta_scale: float = 1.0


def limit_tcp_target(current, desired, duration_s: float, config: ChunkStreamConfig):
    import numpy as np

    current_array = np.asarray(current, dtype=np.float32)
    desired_array = np.asarray(desired, dtype=np.float32).copy()
    if current_array.shape != (6,) or desired_array.shape != (6,):
        raise ValueError("current and desired TCP poses must each contain six values")
    desired_array[3:6] = nearest_rotation_vector(desired_array[3:6], current_array[3:6])
    target = current_array + config.action_delta_scale * (desired_array - current_array)
    translation = target[:3] - current_array[:3]
    rotation = target[3:] - current_array[3:]
    for delta, limit in (
        (translation, config.max_linear_velocity * duration_s),
        (rotation, config.max_angular_velocity * duration_s),
    ):
        norm = float(np.linalg.norm(delta))
        if norm > limit and norm > 1e-9:
            delta *= limit / norm
    return np.concatenate((current_array[:3] + translation, current_array[3:] + rotation)).astype(np.float32)


def interpolated_tcp_targets(start, target, steps: int):
    if steps < 1:
        raise ValueError("steps must be positive")
    return [interpolate_pose(start, target, index, steps) for index in range(1, steps + 1)]


def stream_tcp_target(controller: Any, desired: Sequence[float], config: ChunkStreamConfig) -> list[float]:
    """Velocity-limit one 10 Hz target, then stream a 500 Hz minimum-jerk segment."""
    duration_s = 1.0 / config.policy_hz
    start = controller.get_latest_tcp()
    if start is None:
        raise RuntimeError("RTDE has not produced a TCP pose")
    target = limit_tcp_target(start, desired, duration_s, config)
    steps = max(1, round(config.servo_hz * duration_s))
    deadline = time.monotonic()
    period = duration_s / steps
    for pose in interpolated_tcp_targets(start, target, steps):
        controller.set_target_tcp(pose)
        deadline += period
        wait = deadline - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    return target.tolist()
