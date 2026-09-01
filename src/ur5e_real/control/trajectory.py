from __future__ import annotations

from collections.abc import Sequence


def minimum_jerk(progress: float) -> float:
    """Quintic minimum-jerk blend for a normalized time in [0, 1]."""
    value = max(0.0, min(1.0, float(progress)))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def interpolate_pose(start: Sequence[float], target: Sequence[float], elapsed_s: float, duration_s: float) -> list[float]:
    if len(start) != 6 or len(target) != 6:
        raise ValueError("start and target poses must contain six values")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    blend = minimum_jerk(elapsed_s / duration_s)
    return [float(a) + blend * (float(b) - float(a)) for a, b in zip(start, target)]
