from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

from ..data.schema import nearest_rotation_vector
from ..hardware.urscript import speedl_command
from .chunk import ChunkStreamConfig, limit_tcp_target


@dataclass(frozen=True)
class SocketSpeedLConfig:
    policy_hz: float = 10.0
    smoothing_alpha: float = 0.7
    acceleration: float = 0.5
    max_linear_velocity: float = 0.20
    max_angular_velocity: float = 0.5
    tracking_gain: float = 10.0


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


def tracking_speedl_velocity(
    current: Sequence[float], target: Sequence[float], config: SocketSpeedLConfig
):
    """Calculate a bounded Cartesian velocity that converges on a pose target."""
    import numpy as np

    if config.tracking_gain <= 0:
        raise ValueError("tracking_gain must be positive")
    current_array = np.asarray(current, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32).copy()
    if current_array.shape != (6,) or target_array.shape != (6,):
        raise ValueError("current and target TCP poses must each contain six values")
    target_array[3:6] = nearest_rotation_vector(target_array[3:6], current_array[3:6])
    velocity = config.tracking_gain * (target_array - current_array)
    for component, limit in (
        (velocity[:3], config.max_linear_velocity),
        (velocity[3:], config.max_angular_velocity),
    ):
        norm = float(np.linalg.norm(component))
        if norm > limit and norm > 1e-9:
            component *= limit / norm
    return velocity.astype(np.float32)


class SocketInferenceGapController:
    """Track one final pose while the next policy chunk is being inferred."""

    def __init__(
        self,
        connection: Any,
        read_state: Callable[[], tuple[float, list[float]] | None],
        target: Sequence[float],
        config: SocketSpeedLConfig,
    ) -> None:
        self.connection = connection
        self.read_state = read_state
        self.config = config
        self.target = [float(value) for value in target]
        if len(self.target) != 6:
            raise ValueError("TCP target must contain six values")
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name="ur5e-socket-gap-hold",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.stop()
            raise RuntimeError("timed out starting socket inference-gap control")
        self.check()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("socket inference-gap control failed") from self._error

    def _loop(self) -> None:
        period = 1.0 / self.config.policy_hz
        try:
            while not self._stop.is_set():
                state = self.read_state()
                if state is not None:
                    velocity = tracking_speedl_velocity(state[1], self.target, self.config)
                    self.connection.sendall(
                        speedl_command(
                            velocity,
                            acceleration=self.config.acceleration,
                            duration_s=period,
                        ).encode("utf-8")
                    )
                    self._ready.set()
                self._stop.wait(period)
        except Exception as exc:
            if not self._stop.is_set():
                self._error = exc
            self._ready.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
