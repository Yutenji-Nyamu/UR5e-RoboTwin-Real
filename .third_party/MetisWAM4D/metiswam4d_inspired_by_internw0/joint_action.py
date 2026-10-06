"""InternW0-Delta's RoboDojo action / state representation.

14-D absolute joint vector ``[left arm 6, left gripper, right arm 6, right gripper]`` (grippers in [0, 1], 1 = open),
z-scored with the checkpoint's ``global_mean`` / ``global_std`` (clamped to +-5) and scattered into the unified 80-D
space at ``[0:6] [16] [40:46] [56]``; every other slot is padding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATS_PATH = Path("/m2v_intern_v3/danglingwei/files/InternW0_delta_src/XPolicyLab/policy/InternW0_delta/"
                  "config/dataset_stats.json")
SLOTS = ((0, 6, 0, 6), (6, 7, 16, 17), (7, 13, 40, 46), (13, 14, 56, 57))   # (src0, src1, dst0, dst1)
UNIFIED_DIM = 80
GRIPPERS = (6, 13)
CLAMP = 5.0


class JointNormalizer:
    def __init__(self, path: str | Path = STATS_PATH):
        stats = json.loads(Path(path).read_text())
        self.mean = {k: np.asarray(stats[k]["default"]["global_mean"], np.float32) for k in ("action", "state")}
        self.std = {k: np.asarray(stats[k]["default"]["global_std"], np.float32) for k in ("action", "state")}

    def normalize(self, joints: np.ndarray, kind: str) -> np.ndarray:
        z = (np.asarray(joints, np.float32) - self.mean[kind]) / (self.std[kind] + 1e-8)
        return np.clip(z, -CLAMP, CLAMP).astype(np.float32)

    def denormalize(self, z: np.ndarray, kind: str = "action") -> np.ndarray:
        return (np.asarray(z, np.float32) * (self.std[kind] + 1e-8) + self.mean[kind]).astype(np.float32)


def scatter(z14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[..., 14]`` -> (``[..., 80]`` values, ``[80]`` bool valid-dimension mask)."""
    out = np.zeros((*z14.shape[:-1], UNIFIED_DIM), np.float32)
    valid = np.zeros(UNIFIED_DIM, bool)
    for s0, s1, d0, d1 in SLOTS:
        out[..., d0:d1] = z14[..., s0:s1]
        valid[d0:d1] = True
    return out, valid


def gather(u80: np.ndarray) -> np.ndarray:
    out = np.zeros((*u80.shape[:-1], 14), np.float32)
    for s0, s1, d0, d1 in SLOTS:
        out[..., s0:s1] = u80[..., d0:d1]
    return out


def unified_to_joints(u80: np.ndarray, normalizer: JointNormalizer) -> np.ndarray:
    """Model output ``[T, 80]`` -> executable ``[T, 14]`` joint targets (grippers clipped to [0, 1])."""
    joints = normalizer.denormalize(gather(u80), "action")
    joints[..., list(GRIPPERS)] = np.clip(joints[..., list(GRIPPERS)], 0.0, 1.0)
    return joints


def joints_to_action_dicts(joints: np.ndarray) -> list[dict]:
    """``[T, 14]`` -> RoboDojo ``take_action`` dictionaries (joint control)."""
    return [{"left_arm_joint_state": j[0:6].astype(np.float32), "left_ee_joint_state": j[6:7].astype(np.float32),
             "right_arm_joint_state": j[7:13].astype(np.float32), "right_ee_joint_state": j[13:14].astype(np.float32)}
            for j in np.asarray(joints, np.float32)]


__all__ = ["JointNormalizer", "SLOTS", "gather", "joints_to_action_dicts", "scatter", "unified_to_joints"]
