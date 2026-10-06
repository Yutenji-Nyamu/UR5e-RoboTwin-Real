"""Physical joint7 semantics; the native 80-D vector is only a model wrapper."""

import hashlib
import json

import numpy as np

from ...control.joint import JointMotionConfig, joint_vector

SLOTS = tuple(range(10, 17))
HORIZON = 50
FPS = 10
VERSION = "ur5e_metis_joint7_rgb_v1"
JOINT_ORDER = ("base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3")


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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def decode_actions(actions):
    values = finite(actions, 7)
    if values.shape != (HORIZON, 7) or np.any((values[:, 6] < 0) | (values[:, 6] > 1)):
        raise ValueError("Metis actions must be (50, 7) absolute joint radians + gripper in [0, 1]")
    return values[:, :6].copy(), values[:, 6].copy()


def motion_config(contract, *, speed=0.6):
    return JointMotionConfig(tuple(contract["joint_lower"]), tuple(contract["joint_upper"]),
                             policy_hz=FPS, max_velocity_rad_s=speed)


def validate_runtime_contract(contract):
    expected = {"version": VERSION, "runtime_version": 1, "fps": FPS, "horizon": HORIZON,
                "slots": list(SLOTS), "joint_order": list(JOINT_ORDER), "joint_unit": "rad",
                "wire_action": "absolute_joint7", "stop": "bounded_chunks"}
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"Metis contract mismatch: {name} must be {value!r}")
    if not contract.get("task") or not contract.get("dataset_id"):
        raise ValueError("task and dataset identity required")
    if contract.get("initial_gripper") not in (0, 1):
        raise ValueError("initial gripper must be 0 or 1")
    motion_config(contract).check(contract["home_q"])
    joint_vector(contract["tcp_offset"], name="TCP offset")
    low, high = np.asarray(contract["tcp_lower"]), np.asarray(contract["tcp_upper"])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("invalid TCP monitoring envelope")
    for kind in ("state", "action"):
        normalize(np.zeros(7), contract["stats"], kind)
    digest(contract)
    return contract


def require_same_contract(expected, actual):
    validate_runtime_contract(expected)
    validate_runtime_contract(actual)
    if digest(expected) != digest(actual):
        raise ValueError("Metis server and local dataset contracts differ")


def require_home(q, contract, *, tolerance_rad=0.03):
    if np.max(np.abs(joint_vector(q) - joint_vector(contract["home_q"]))) > tolerance_rad:
        raise RuntimeError("joint start posture differs from the dataset; run Metis home first")
