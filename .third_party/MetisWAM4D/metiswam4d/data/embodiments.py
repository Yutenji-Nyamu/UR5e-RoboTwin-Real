"""Unified action interface across embodiments.

Actions live in OpenWAM's 80-D unified space.  Each embodiment declares how its
raw action vector scatters into that space (``unify_map``, closed ranges) and
how it is normalised.  The canonical layout inherited from OpenWAM-Alpha is one
10-D end-effector block per arm

    left arm  slots  0- 9 : xyz (3) + rot6d (6) + gripper (1)
    right arm slots 34-43

which is what all Alpha checkpoints were trained on.  Joint-space blocks are
offered as an experimental alternative (slots 10-23 / 44-57, not used by the
Alpha pretraining).  ``dim_mask`` marks the valid unified slots and enters the
action loss so unused slots are never supervised.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

UNIFY_DIM = 80
EEF_LEFT = "0-9"
EEF_RIGHT = "34-43"
JOINT_LEFT = "10-23"     # experimental: up to 7 joints + gripper per arm
JOINT_RIGHT = "44-57"


def _expand(token) -> list[int]:
    if isinstance(token, (list, tuple)):
        out: list[int] = []
        for t in token:
            out.extend(_expand(t))
        return out
    if isinstance(token, (int, np.integer)):
        return [int(token)]
    s = str(token).strip()
    if "-" in s:
        a, b = (int(x) for x in s.split("-", 1))
        if b < a:
            raise ValueError(f"bad range {token!r}")
        return list(range(a, b + 1))
    return [int(s)]


def parse_unify_spec(spec: Sequence, unify_dim: int = UNIFY_DIM) -> np.ndarray:
    """Single-list spec (source dims implicit 0..N-1) -> ``dst_index [N]``."""
    dst = np.asarray(_expand(list(spec)), dtype=np.int64)
    if dst.size == 0 or dst.min() < 0 or dst.max() >= unify_dim:
        raise ValueError("unify spec out of range")
    if len(np.unique(dst)) != len(dst):
        raise ValueError("unify spec maps two source dims to one slot")
    return dst


def map_to_unify(action: np.ndarray, dst_index: np.ndarray, unify_dim: int = UNIFY_DIM):
    action = np.asarray(action)
    if action.shape[-1] != dst_index.shape[0]:
        raise ValueError(f"action width {action.shape[-1]} != mapping size {dst_index.shape[0]}")
    unified = np.zeros((*action.shape[:-1], unify_dim), dtype=action.dtype)
    unified[..., dst_index] = action
    mask = np.zeros(unify_dim, dtype=bool)
    mask[dst_index] = True
    return unified, mask


def unmap_from_unify(unified: np.ndarray, dst_index: np.ndarray) -> np.ndarray:
    return np.asarray(unified)[..., dst_index]


@dataclass(frozen=True)
class Normalizer:
    """Per-dimension affine map of raw actions to [-1, 1]."""
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def from_stats(cls, stats: dict, mode: str = "min-max") -> "Normalizer":
        if mode == "min-max":
            low, high = stats["min"], stats["max"]
        elif mode == "quantile":
            low, high = stats["q01"], stats["q99"]
        else:
            raise ValueError(f"unknown normalisation mode {mode!r}")
        return cls(np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        span = np.maximum(self.high - self.low, 1e-6)
        return np.clip(2.0 * (np.asarray(raw, dtype=np.float32) - self.low) / span - 1.0, -1.0, 1.0)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        span = np.maximum(self.high - self.low, 1e-6)
        return (np.asarray(value, dtype=np.float32) + 1.0) / 2.0 * span + self.low


@dataclass(frozen=True)
class Embodiment:
    name: str
    raw_dim: int
    unify_map: tuple[str, ...]
    mode: str = "eef"                    # "eef" or "joint"
    normalize_mode: str = "min-max"
    stats_path: str | None = None        # .npy dict with min/max/q01/q99 per raw dim
    description: str = ""

    @property
    def dst_index(self) -> np.ndarray:
        if self.raw_dim == 0:
            return np.zeros(0, dtype=np.int64)
        dst = parse_unify_spec(self.unify_map)
        if dst.shape[0] != self.raw_dim:
            raise ValueError(f"{self.name}: unify map covers {dst.shape[0]} dims, raw_dim is {self.raw_dim}")
        return dst

    @property
    def dim_mask(self) -> np.ndarray:
        mask = np.zeros(UNIFY_DIM, dtype=bool)
        mask[self.dst_index] = True
        return mask

    def normalizer(self) -> Normalizer | None:
        if self.stats_path is None:
            return None
        stats = np.load(Path(self.stats_path), allow_pickle=True)
        stats = stats.item() if isinstance(stats, np.ndarray) and stats.dtype == object else stats
        if self.mode in stats:
            stats = stats[self.mode]
        return Normalizer.from_stats(stats, self.normalize_mode)

    def encode(self, raw_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw ``[..., raw_dim]`` -> (unified normalised ``[..., 80]``, dim mask ``[80]``)."""
        norm = self.normalizer()
        value = norm.normalize(raw_action) if norm is not None else np.asarray(raw_action, dtype=np.float32)
        return map_to_unify(value, self.dst_index)

    def decode(self, unified: np.ndarray) -> np.ndarray:
        raw = unmap_from_unify(unified, self.dst_index)
        norm = self.normalizer()
        return norm.denormalize(raw) if norm is not None else raw


class EmbodimentRegistry:
    def __init__(self, embodiments: Sequence[Embodiment]):
        self._by_name = {e.name: e for e in embodiments}
        self._order = [e.name for e in embodiments]

    def __getitem__(self, name: str) -> Embodiment:
        return self._by_name[name]

    def index(self, name: str) -> int:
        return self._order.index(name)

    def name(self, index: int) -> str:
        return self._order[index]

    def names(self) -> list[str]:
        return list(self._order)

    def with_stats(self, name: str, stats_path: str) -> "EmbodimentRegistry":
        items = [e if e.name != name else dataclasses.replace(e, stats_path=stats_path)
                 for e in (self._by_name[n] for n in self._order)]
        return EmbodimentRegistry(items)


def default_registry() -> EmbodimentRegistry:
    eef = (EEF_LEFT, EEF_RIGHT)
    return EmbodimentRegistry([
        Embodiment("human_ego", 0, (), mode="none", description="human egocentric video, no action"),
        Embodiment("robodojo_arx_x5", 20, eef, description="RoboDojo sim, ARX X5 dual arm, per-arm EEF10"),
        Embodiment("robotwin2_aloha_agilex", 20, eef, description="RoboTwin 2.0 aloha-agilex, per-arm EEF10"),
        Embodiment("ig10k_realman", 20, eef, description="IG-10K real dual-arm Realman: ee_pose(12)+gripper(2) -> per-arm EEF10"),
        Embodiment("ig10k_sim_franka", 20, eef, description="IG-10K ManiSkill3 dual Franka: FK to per-arm EEF10"),
        Embodiment("ur5e_dual_real", 20, eef, description="Real UR5e dual arm (UR5e-RoboTwin-Real), FK to per-arm EEF10"),
        Embodiment("ur5e_dual_joint", 14, ("10-16", "44-50"), mode="joint",
                   description="experimental joint-space interface: 6 joints + gripper per arm"),
        Embodiment("ebench_lift2", 23, (EEF_LEFT, EEF_RIGHT, "68-70"),
                   description="EBench (GenManip) lift2 mobile dual arm: per-arm EEF10 + base dx/dy/dyaw"),
        Embodiment("vlabench_franka", 10, (EEF_LEFT,),
                   description="VLABench (MuJoCo) single Franka Panda: EEF10 in the robot base frame"),
    ])


__all__ = [
    "EEF_LEFT", "EEF_RIGHT", "Embodiment", "EmbodimentRegistry", "JOINT_LEFT", "JOINT_RIGHT", "Normalizer",
    "UNIFY_DIM", "default_registry", "map_to_unify", "parse_unify_spec", "unmap_from_unify",
]
