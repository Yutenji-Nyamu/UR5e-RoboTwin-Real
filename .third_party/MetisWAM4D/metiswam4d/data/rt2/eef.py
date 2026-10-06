"""RoboTwin end-effector actions in OpenWAM-Alpha's convention.

EEF20 = [left xyz(3), left rot6d(6), left gripper(1), right xyz(3), right rot6d(6), right gripper(1)].
RoboTwin ``endpose`` quaternions are xyzw; rot6d = first two columns of the rotation matrix.
Normalisation: min-max to [-1, 1] with the checkpoint's ``normalization_stats.npy['eef']`` (Alpha's
training-side formula ``2 (x - lo) / (hi - lo) - 1``).  Unified 80-D slots: left 0-9, right 34-43.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

EEF_DIM = 20
UNIFY_DIM = 80
SLOTS = np.concatenate((np.arange(0, 10), np.arange(34, 44)))


def quat_xyzw_to_rot6d(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    mat = Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()  # scipy takes xyzw
    return np.concatenate((mat[..., :, 0], mat[..., :, 1]), axis=-1).astype(np.float32)


def read_eef20(source: h5py.File, frames: np.ndarray) -> np.ndarray:
    """``[len(frames), 20]`` raw EEF for the given frame indices."""
    idx = np.asarray(frames, dtype=np.int64)
    arms = []
    for side in ("left", "right"):
        pose = source[f"endpose/{side}_endpose"][idx].astype(np.float32)
        grip = source[f"endpose/{side}_gripper"][idx].astype(np.float32)
        arms.append(np.concatenate((pose[:, :3], quat_xyzw_to_rot6d(pose[:, 3:7]), grip[:, None]), axis=-1))
    return np.concatenate(arms, axis=-1).astype(np.float32)


def load_pickled_stats(path: str | Path) -> dict:
    """``np.load(..., allow_pickle=True).item()`` that also reads numpy>=2 pickles under numpy 1.x
    (``numpy._core`` -> ``numpy.core`` module rename)."""
    try:
        return np.load(Path(path), allow_pickle=True).item()
    except ModuleNotFoundError as err:
        if "numpy._core" not in str(err):
            raise
    import pickle

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        np.lib.format._read_array_header(f, version)
        return _Unpickler(f).load().item()


class EEFNormalizer:
    def __init__(self, stats_path: str | Path, eps: float = 1e-6):
        stats = load_pickled_stats(stats_path)
        eef = stats["eef"]
        self.lo = np.asarray(eef["min"], dtype=np.float32)
        self.hi = np.asarray(eef["max"], dtype=np.float32)
        if self.lo.shape != (EEF_DIM,):
            raise ValueError(f"expected 20-D EEF stats, got {self.lo.shape}")
        self.range = np.maximum(self.hi - self.lo, eps)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return (2.0 * (np.asarray(raw, dtype=np.float32) - self.lo) / self.range - 1.0).astype(np.float32)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return ((np.asarray(value, dtype=np.float32) + 1.0) / 2.0 * self.range + self.lo).astype(np.float32)


def scatter_unified(eef20: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((*eef20.shape[:-1], UNIFY_DIM), dtype=np.float32)
    out[..., SLOTS] = eef20
    mask = np.zeros(UNIFY_DIM, dtype=bool)
    mask[SLOTS] = True
    return out, mask


def gather_unified(unified: np.ndarray) -> np.ndarray:
    return np.asarray(unified)[..., SLOTS]


__all__ = ["EEF_DIM", "EEFNormalizer", "SLOTS", "UNIFY_DIM", "gather_unified", "quat_xyzw_to_rot6d", "read_eef20", "scatter_unified"]
