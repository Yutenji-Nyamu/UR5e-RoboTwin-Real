from __future__ import annotations

import time
from collections.abc import Callable
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


def continuous_chunk_targets(start, desired_poses, config: ChunkStreamConfig):
    """Build one continuous 500 Hz reference without stopping at 10 Hz knots."""
    import numpy as np

    previous = np.asarray(start, dtype=np.float32)
    if previous.shape != (6,):
        raise ValueError("start TCP pose must contain six values")
    duration_s = 1.0 / config.policy_hz
    steps = max(1, round(config.servo_hz * duration_s))
    samples = []
    waypoints = []
    for desired in desired_poses:
        target = limit_tcp_target(previous, desired, duration_s, config)
        # A linear 500 Hz reference remains in motion across adjacent policy
        # knots; servoJ's lookahead smooths changes in direction. In contrast,
        # a separate minimum-jerk segment forces zero velocity at every knot.
        for index in range(1, steps + 1):
            alpha = index / steps
            samples.append((previous + alpha * (target - previous)).astype(np.float32))
        waypoints.append(target)
        previous = target
    return samples, waypoints, steps


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


def stream_tcp_chunk(
    controller: Any,
    desired_poses,
    config: ChunkStreamConfig,
    *,
    on_waypoint: Callable[[int, list[float]], None] | None = None,
) -> list[list[float]]:
    """Stream a complete policy chunk on one clock with no per-action pause."""
    start = controller.get_latest_tcp()
    if start is None:
        raise RuntimeError("RTDE has not produced a TCP pose")
    samples, waypoints, steps_per_action = continuous_chunk_targets(start, desired_poses, config)
    period = 1.0 / config.servo_hz
    deadline = time.monotonic()
    for sample_index, pose in enumerate(samples, start=1):
        controller.set_target_tcp(pose)
        deadline += period
        wait = deadline - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if on_waypoint is not None and sample_index % steps_per_action == 0:
            waypoint_index = sample_index // steps_per_action - 1
            on_waypoint(waypoint_index, waypoints[waypoint_index].tolist())
    return [target.tolist() for target in waypoints]
