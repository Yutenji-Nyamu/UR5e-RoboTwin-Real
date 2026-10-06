"""Physical joint7 semantics; the native 80-D vector is only a model wrapper."""

import numpy as np

SLOTS = tuple(range(10, 17))
HORIZON = 50
FPS = 10
VERSION = "ur5e_metis_joint7_rgb_v1"


def finite(values, width):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != width or not np.isfinite(values).all():
        raise ValueError(f"expected finite values with final dimension {width}")
    return values


def to_unified(values):
    values = finite(values, 7)
    out = np.zeros((*values.shape[:-1], 80), dtype=np.float32)
    out[..., SLOTS] = values
    return out


def to_pi05(values):
    values = finite(values, 7)
    out = np.zeros((*values.shape[:-1], 14), dtype=np.float32)
    out[..., :6] = out[..., 7:13] = values[..., :6]
    out[..., 13] = np.clip(values[..., 6], 0, 1)
    return out


def delta_actions(state, actions):
    state, result = finite(state, 7), finite(actions, 7).copy()
    result[..., :6] -= state[..., None, :6]
    return result


def absolute_actions(state, actions):
    state, result = finite(state, 7), finite(actions, 7).copy()
    result[..., :6] += state[..., None, :6]
    result[..., 6] = np.clip(result[..., 6], 0, 1)
    return result


def fit_stats(states, actions, valid):
    """Only the explicitly selected training episodes contribute statistics."""
    states, actions = finite(states, 7), delta_actions(states, actions)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != actions.shape[:-1] or not valid.any():
        raise ValueError("action validity must cover the training time steps")
    return {
        "state_mean": states.mean(axis=0).tolist(),
        "state_std": np.maximum(states.std(axis=0), 1e-3).tolist(),
        "action_mean": actions[valid].mean(axis=0).tolist(),
        "action_std": np.maximum(actions[valid].std(axis=0), 1e-3).tolist(),
    }


def normalize(values, stats, kind, *, inverse=False):
    values = finite(values, 7)
    mean, std = finite(stats[f"{kind}_mean"], 7), finite(stats[f"{kind}_std"], 7)
    if mean.shape != (7,) or std.shape != (7,) or np.any(std <= 0):
        raise ValueError("normalization requires seven finite means and positive scales")
    return values * std + mean if inverse else (values - mean) / std
