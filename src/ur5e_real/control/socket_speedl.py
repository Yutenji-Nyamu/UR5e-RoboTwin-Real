from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .chunk import ChunkStreamConfig, limit_tcp_target


@dataclass(frozen=True)
class SocketSpeedLConfig:
    policy_hz: float = 10.0
    smoothing_alpha: float = 0.7
    acceleration: float = 0.5
    max_linear_velocity: float = 0.20
    max_angular_velocity: float = 0.5


def smoothed_speedl_target(
    current: Sequence[float],
    desired: Sequence[float],
    previous_target: Sequence[float] | None,
    config: SocketSpeedLConfig,
):
    """Apply the historical target EMA, then convert it to a bounded speedL command."""
    import numpy as np

    if not 0.0 < config.smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be in (0, 1]")
    if config.max_linear_velocity <= 0:
        raise ValueError("max_linear_velocity must be positive")
    duration_s = 1.0 / config.policy_hz
    limits = ChunkStreamConfig(
        policy_hz=config.policy_hz,
        max_linear_velocity=config.max_linear_velocity,
        max_angular_velocity=config.max_angular_velocity,
    )
    current_array = np.asarray(current, dtype=np.float32)
    bounded = limit_tcp_target(current_array, desired, duration_s, limits)
    base = current_array if previous_target is None else np.asarray(previous_target, dtype=np.float32)
    smoothed = base + config.smoothing_alpha * (bounded - base)
    smoothed = limit_tcp_target(current_array, smoothed, duration_s, limits)
    velocity = (smoothed - current_array) / duration_s
    return smoothed.astype(np.float32), velocity.astype(np.float32)
