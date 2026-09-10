"""Canonical 7D bounded coordinates; native quantile/delta decode exactly once."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ACTION_VERSION

INDICES = (0, 1, 2, 3, 4, 5, 13)


@dataclass(frozen=True)
class ActionCodec:
    scale: np.ndarray
    q01: np.ndarray
    q99: np.ndarray

    def __post_init__(self):
        for name in ("scale", "q01", "q99"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain 7 finite values")
            object.__setattr__(self, name, value)
        if (self.scale < 1).any() or (self.q99 < self.q01).any():
            raise ValueError("invalid canonical action range")

    @classmethod
    def fit(cls, raw_references, norm_stats):
        values = np.asarray(raw_references)[..., INDICES]
        if not np.isfinite(values).all() or values.size == 0:
            raise ValueError("cannot fit action coordinates without finite reference actions")
        scale = np.maximum(1.25, np.max(np.abs(values.reshape(-1, 7)), axis=0) * 1.25)
        stats = norm_stats.get("norm_stats", norm_stats)["actions"]
        return cls(scale, np.asarray(stats["q01"])[list(INDICES)], np.asarray(stats["q99"])[list(INDICES)])

    def encode(self, raw_reference):
        result = np.asarray(raw_reference, dtype=np.float64)[..., INDICES] / self.scale
        if not np.isfinite(result).all() or (np.abs(result) >= 1).any():
            raise ValueError(
                "reference exceeds the versioned action range; refit/review the codec, do not silently clip"
            )
        return result.astype(np.float32)

    def decode(self, actions, observation_q):
        value = np.asarray(actions, dtype=np.float64)
        q = np.asarray(observation_q, dtype=np.float64)
        if value.shape[-1] != 7 or q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("decode requires canonical 7D actions and the same observation's q6")
        if not np.isfinite(value).all() or (np.abs(value) > 1 + 1e-6).any():
            raise ValueError("canonical action is not finite/bounded")
        decoded = (value * self.scale + 1) / 2 * (self.q99 - self.q01 + 1e-6) + self.q01
        decoded[..., :6] += q  # restore absolute joints using observation, not a later measured state
        decoded[..., 6] = np.clip(decoded[..., 6], 0, 1)
        return decoded

    def as_dict(self):
        return {"version": ACTION_VERSION, **{k: getattr(self, k).tolist() for k in ("scale", "q01", "q99")}}

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != ACTION_VERSION:
            raise ValueError("canonical action contract mismatch")
        return cls(**{k: value[k] for k in ("scale", "q01", "q99")})


def proprio(normalized_state, executor_aux=None):
    state = np.asarray(normalized_state, dtype=np.float32)[list(INDICES)]
    aux = np.zeros(6, np.float32) if executor_aux is None else np.asarray(executor_aux, dtype=np.float32)
    if aux.shape != (6,) or not np.isfinite(aux).all() or not np.isfinite(state).all():
        raise ValueError("invalid proprioception/executor state")
    return np.concatenate([state, aux])
