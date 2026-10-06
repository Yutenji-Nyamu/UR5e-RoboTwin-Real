"""ARX X5 end-effector actions in OpenWAM-Alpha-Sim-RoboDojo's convention.

RDJ_MetisWAM4D stores ``eef20`` in the world frame (link6 pose from the official ``state/*_ee_poses``).
Alpha's RoboDojo checkpoint was trained on **per-arm robot-base** poses (OpenWAM ``robodojo_contract``:
bases at ``(-/+0.3, -0.45, 0.765)``, base quaternion wxyz ``(0.707, 0, 0, 0.707)`` = 90 deg yaw), normalised
min-max with its ``normalization_stats.npy``.  Position ``p_b = R_b^T (p_w - t_b)``; the rot6d columns are the
first two columns of ``R_b^T R_w``.  The gripper (0 open .. 1 closed) is unchanged.
"""
from __future__ import annotations

import numpy as np

LEFT_BASE_POS = np.asarray((-0.3, -0.45, 0.765), dtype=np.float64)
RIGHT_BASE_POS = np.asarray((0.3, -0.45, 0.765), dtype=np.float64)
BASE_QUAT_WXYZ = np.asarray((0.707, 0.0, 0.0, 0.707), dtype=np.float64)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (np.asarray(q, dtype=np.float64) / np.linalg.norm(q)).tolist()
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


BASE_ROTATION = quat_wxyz_to_matrix(BASE_QUAT_WXYZ)   # world <- base


def _arm_world_to_base(arm10: np.ndarray, base_pos: np.ndarray) -> np.ndarray:
    """``[..., 10]`` xyz + rot6d + gripper, world -> robot base."""
    rt = BASE_ROTATION.T
    pos = (arm10[..., :3] - base_pos) @ rt.T
    c1 = arm10[..., 3:6] @ rt.T
    c2 = arm10[..., 6:9] @ rt.T
    return np.concatenate((pos, c1, c2, arm10[..., 9:10]), axis=-1)


def world_to_base_eef20(eef20_world: np.ndarray) -> np.ndarray:
    """``[..., 20]`` world-frame EEF20 -> Alpha RoboDojo's per-arm base-frame EEF20 (float32)."""
    e = np.asarray(eef20_world, dtype=np.float64)
    if e.shape[-1] != 20:
        raise ValueError(f"expected EEF20, got {e.shape}")
    left = _arm_world_to_base(e[..., :10], LEFT_BASE_POS)
    right = _arm_world_to_base(e[..., 10:], RIGHT_BASE_POS)
    return np.concatenate((left, right), axis=-1).astype(np.float32)


def base_to_world_eef20(eef20_base: np.ndarray) -> np.ndarray:
    """Inverse of :func:`world_to_base_eef20`."""
    e = np.asarray(eef20_base, dtype=np.float64)
    out = []
    for arm, base_pos in ((e[..., :10], LEFT_BASE_POS), (e[..., 10:], RIGHT_BASE_POS)):
        pos = arm[..., :3] @ BASE_ROTATION.T + base_pos
        c1 = arm[..., 3:6] @ BASE_ROTATION.T
        c2 = arm[..., 6:9] @ BASE_ROTATION.T
        out.append(np.concatenate((pos, c1, c2, arm[..., 9:10]), axis=-1))
    return np.concatenate(out, axis=-1).astype(np.float32)


__all__ = ["BASE_QUAT_WXYZ", "BASE_ROTATION", "LEFT_BASE_POS", "RIGHT_BASE_POS", "base_to_world_eef20",
           "quat_wxyz_to_matrix", "world_to_base_eef20"]
