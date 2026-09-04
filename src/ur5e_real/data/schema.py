from __future__ import annotations

import math
from typing import Sequence


SCHEMA_VERSION = 2
ROBOTWIN_ACTION_DIM = 14

ACTION_COLUMNS = [
    "controller_time_s",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "gripper_state",
    "gripper_event_counter",
]

# RoboTwin ACT expects three camera keys. The current workcell has two physical
# cameras, so the left wrist observation intentionally duplicates the wrist view.
CAMERA_GROUP_MAP = {
    "head_camera": "head",
    "right_camera": "wrist",
    "left_camera": "wrist",
}


def nearest_rotation_vector(rotation: Sequence[float], reference: Sequence[float]) -> list[float]:
    """Return the equivalent axis-angle branch closest to ``reference``."""
    values = [float(value) for value in rotation]
    norm = math.sqrt(sum(value * value for value in values))
    candidates = [values]
    if norm > 1e-9:
        unit = [value / norm for value in values]
        candidates.extend(
            [values[index] + turn * 2.0 * math.pi * unit[index] for index in range(3)]
            for turn in (-1.0, 1.0)
        )
    return min(candidates, key=lambda item: sum((item[index] - reference[index]) ** 2 for index in range(3)))


def continuous_tcp_poses(poses):
    """Copy TCP poses while removing equivalent rotvec branch jumps."""
    import numpy as np

    result = np.asarray(poses, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 6:
        raise ValueError("TCP poses must have shape (N, 6)")
    for index in range(1, len(result)):
        result[index, 3:6] = nearest_rotation_vector(result[index, 3:6], result[index - 1, 3:6])
    return result


def tcp_gripper_vectors(poses, gripper):
    """Encode one real UR5e as RoboTwin's 14-dimensional dual-arm vector."""
    import numpy as np

    tcp = continuous_tcp_poses(poses)
    grip = np.asarray(gripper, dtype=np.float32).reshape(-1, 1)
    if len(tcp) != len(grip):
        raise ValueError("TCP and gripper arrays have different lengths")
    zeros = np.zeros_like(grip)
    return np.concatenate((tcp, zeros, tcp, grip), axis=1).astype(np.float32)
