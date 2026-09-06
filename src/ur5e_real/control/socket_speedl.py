from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..data.schema import nearest_rotation_vector
from .chunk import ChunkStreamConfig, limit_tcp_target

SOCKET_STRETCH_MARGIN_S = 0.02


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


def stretched_speedl_velocity(
    current: Sequence[float],
    target: Sequence[float],
    inference_s: float,
    config: SocketSpeedLConfig,
    *,
    margin_s: float = SOCKET_STRETCH_MARGIN_S,
):
    """Spread the final action over its period plus the expected inference gap."""
    import numpy as np

    if inference_s < 0 or margin_s < 0:
        raise ValueError("inference_s and margin_s must be non-negative")
    current_array = np.asarray(current, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32).copy()
    if current_array.shape != (6,) or target_array.shape != (6,):
        raise ValueError("current and target TCP poses must each contain six values")
    target_array[3:6] = nearest_rotation_vector(target_array[3:6], current_array[3:6])
    duration_s = 1.0 / config.policy_hz + inference_s + margin_s
    velocity = (target_array - current_array) / duration_s
    return velocity.astype(np.float32), duration_s
