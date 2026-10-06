"""IG-10K Realman dual-arm actions in OpenWAM-Alpha's EEF20 convention.

LeRobot columns (``meta/info.json``): ``*.eepos_gripper_*`` is 14-D
``[right x y z rx ry rz, left x y z rx ry rz, right gripper, left gripper]``; the rotation is a
roll-pitch-yaw Euler triple, ``R = Rz(rz) Ry(ry) Rx(rx)`` (scipy ``from_euler("xyz")``).  The
convention was verified on the data: at the 14 wrap-around jumps of rx / rz through +-pi only this
order gives a continuous rot6d (max jump 0.10 vs 1.7-1.9 for the other orders and for a rotation
vector), and |ry| never exceeds pi/2.  ``action.*[t] == observation.*[t + 1]`` exactly, so the
action chunk of a window starting at ``s`` is the action rows ``s .. s+31`` and the proprio is the
observation row ``s`` (RoboTwin uses the same "next state" semantics).

EEF20 = [left xyz(3), left rot6d(6), left gripper(1), right xyz(3), right rot6d(6), right gripper(1)];
rot6d = first two columns of the rotation matrix; normalisation min-max to [-1, 1] with dataset
statistics (``action_stats_eef20.json``, written by ``scripts/data_prep/ig10k_robot_action_stats.py``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metiswam4d.data.rt2.eef import EEF_DIM, scatter_unified

REALMAN_DIM = 14
ACTION_COLUMN = "action.eepos_gripper_actions"
STATE_COLUMN = "observation.eepos_gripper_states"


def euler_xyz_to_rot6d(rpy: np.ndarray) -> np.ndarray:
    """``[..., 3]`` (rx, ry, rz) -> ``[..., 6]``, R = Rz(rz) Ry(ry) Rx(rx)."""
    from scipy.spatial.transform import Rotation
    shape = rpy.shape[:-1]
    mat = Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64).reshape(-1, 3)).as_matrix()
    six = np.concatenate((mat[:, :, 0], mat[:, :, 1]), axis=-1)
    return six.reshape(*shape, 6).astype(np.float32)


def rot6d_to_euler_xyz(six: np.ndarray) -> np.ndarray:
    """Inverse of :func:`euler_xyz_to_rot6d` (Gram-Schmidt on the two columns)."""
    from scipy.spatial.transform import Rotation
    shape = six.shape[:-1]
    a, b = np.asarray(six, dtype=np.float64).reshape(-1, 2, 3).transpose(1, 0, 2)
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (b * c0).sum(-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c2 = np.cross(c0, c1)
    mat = np.stack((c0, c1, c2), axis=-1)
    return Rotation.from_matrix(mat).as_euler("xyz").reshape(*shape, 3).astype(np.float32)


def realman_to_eef20(raw14: np.ndarray) -> np.ndarray:
    """``[..., 14]`` Realman eepos+gripper -> ``[..., 20]`` Alpha EEF20 (left arm first)."""
    raw14 = np.asarray(raw14, dtype=np.float32)
    if raw14.shape[-1] != REALMAN_DIM:
        raise ValueError(f"expected {REALMAN_DIM}-D Realman vector, got {raw14.shape[-1]}")
    right, left = raw14[..., 0:6], raw14[..., 6:12]
    grip_r, grip_l = raw14[..., 12:13], raw14[..., 13:14]
    arm_l = np.concatenate((left[..., :3], euler_xyz_to_rot6d(left[..., 3:6]), grip_l), axis=-1)
    arm_r = np.concatenate((right[..., :3], euler_xyz_to_rot6d(right[..., 3:6]), grip_r), axis=-1)
    return np.concatenate((arm_l, arm_r), axis=-1).astype(np.float32)


def eef20_to_realman(eef20: np.ndarray) -> np.ndarray:
    eef20 = np.asarray(eef20, dtype=np.float32)
    left, right = eef20[..., :10], eef20[..., 10:]
    out = np.concatenate((
        right[..., :3], rot6d_to_euler_xyz(right[..., 3:9]),
        left[..., :3], rot6d_to_euler_xyz(left[..., 3:9]),
        right[..., 9:10], left[..., 9:10]), axis=-1)
    return out.astype(np.float32)


class IG10KActionNormalizer:
    """Min-max to [-1, 1] per EEF20 dim from ``action_stats_eef20.json`` (``{"min": [20], "max": [20]}``)."""

    def __init__(self, stats_path: str | Path, eps: float = 1e-6):
        stats = json.loads(Path(stats_path).read_text())
        self.lo = np.asarray(stats["min"], dtype=np.float32)
        self.hi = np.asarray(stats["max"], dtype=np.float32)
        if self.lo.shape != (EEF_DIM,):
            raise ValueError(f"expected {EEF_DIM}-D stats, got {self.lo.shape}")
        self.range = np.maximum(self.hi - self.lo, eps)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (np.asarray(raw, dtype=np.float32) - self.lo) / self.range - 1.0, -1.0, 1.0).astype(np.float32)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return ((np.asarray(value, dtype=np.float32) + 1.0) / 2.0 * self.range + self.lo).astype(np.float32)


GRIPPER_COLUMNS = (12, 13)
GRIPPER_RAMP_HALF_WIDTH = 2   # steps (30 Hz): a 0/1 step becomes a 5-sample linear ramp 0, .2, .4, .6, .8, 1


def ramp_gripper_steps(gripper: np.ndarray, half_width: int = GRIPPER_RAMP_HALF_WIDTH) -> np.ndarray:
    """``[T, G]`` binary open/close commands -> the same sequence with every step replaced by a linear ramp
    spanning ``+-half_width`` samples (moving average, edge-padded).

    The Realman gripper command is 0/1.  Under an MSE / flow-matching target a step whose timing is uncertain
    by one or two samples costs (2)^2 per sample in the normalised [-1, 1] space and dominated the action loss
    (step 3600: 47% of it from these 2 of 20 dims, 5x higher on windows containing a step).  The ramp keeps the
    sign (threshold 0.5 -> same command at inference) and the step location (ramp centre) while making the target
    smooth over the operator's timing jitter.  Plateaus are untouched.
    """
    g = np.asarray(gripper, dtype=np.float32)
    if half_width <= 0 or g.shape[0] < 2:
        return g
    k = 2 * half_width + 1
    padded = np.concatenate((np.repeat(g[:1], half_width, axis=0), g, np.repeat(g[-1:], half_width, axis=0)), axis=0)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(g.shape[1])], axis=1).astype(np.float32)


def encode_window_actions(states14: np.ndarray, normalizer: IG10KActionNormalizer,
                          gripper_ramp: int = GRIPPER_RAMP_HALF_WIDTH):
    """``[H + 1, 14]`` (proprio row followed by H action rows) -> unified ``(action [H, 80], action_mask [H, 80],
    proprio [1, 80], proprio_mask [1, 80])``.  The action rows' gripper columns are ramped (see
    :func:`ramp_gripper_steps`); the proprio row keeps the observed 0/1 state."""
    states14 = np.asarray(states14, dtype=np.float32).copy()
    if gripper_ramp > 0:
        cols = list(GRIPPER_COLUMNS)
        # ramp over the whole window (proprio included as the left context) but only write the action rows back
        states14[1:, cols] = ramp_gripper_steps(states14[:, cols], gripper_ramp)[1:]
    unified, dims = scatter_unified(normalizer.normalize(realman_to_eef20(states14)))
    h = unified.shape[0] - 1
    action_mask = np.broadcast_to(dims, (h, unified.shape[-1])).copy()
    return unified[1:], action_mask, unified[:1], dims[None].copy()


__all__ = ["ACTION_COLUMN", "GRIPPER_COLUMNS", "GRIPPER_RAMP_HALF_WIDTH", "IG10KActionNormalizer", "REALMAN_DIM",
           "STATE_COLUMN", "encode_window_actions", "eef20_to_realman", "euler_xyz_to_rot6d", "ramp_gripper_steps",
           "realman_to_eef20", "rot6d_to_euler_xyz"]
