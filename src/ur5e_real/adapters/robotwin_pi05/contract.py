from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ...control.joint import JointMotionConfig, joint_vector

ROBOTWIN_COMMIT = "210720340637cb4619283b295dde4cdd807c9e66"
NATIVE_PATCH = "ur5e_pi05_config_import_v1"
STATE_LAYOUT = "ur5e_joint14_v1"
JOINT_ORDER = ("base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3")
PROMPT = "Pick up the cube and place it back on the table."


def encode_state(q, gripper: float) -> np.ndarray:
    joints = joint_vector(q)
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("gripper state must be finite in [0, 1]")
    return np.concatenate([joints, [0.0], joints, [gripper]]).astype(np.float32)


def decode_actions(actions, *, horizon: int = 50):
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != (horizon, 14) or not np.isfinite(values).all():
        raise ValueError(f"pi05 actions must be finite ({horizon}, 14) absolute joint/gripper targets")
    # Native quantile outputs may be slightly outside [0, 1]; only the gripper is clipped.
    return values[:, :6].copy(), np.clip(values[:, 13], 0, 1)


def validate_contract(contract: dict) -> dict:
    expected = {
        "version": 1,
        "policy": "pi05",
        "action_space": "joint_position",
        "state_layout": STATE_LAYOUT,
        "joint_unit": "rad",
        "fps": 10,
        "action_horizon": 50,
        "adapt_to_pi": False,
        "delta_joint_actions": True,
        "robotwin_commit": ROBOTWIN_COMMIT,
        "native_patch": NATIVE_PATCH,
    }
    for name, value in expected.items():
        if name not in contract or contract[name] != value:
            raise ValueError(f"pi05 contract mismatch: {name} must be {value!r}")
    if contract.get("joint_order") != list(JOINT_ORDER):
        raise ValueError("pi05 joint order mismatch")
    motion = motion_config(contract)
    motion.check(contract["home_q"])
    joint_vector(contract["tcp_offset"], name="TCP offset")
    low, high = np.asarray(contract["tcp_lower"]), np.asarray(contract["tcp_upper"])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("invalid TCP monitoring envelope")
    if not isinstance(contract.get("dataset_id"), str) or not contract["dataset_id"]:
        raise ValueError("dataset identity is required")
    return contract


def motion_config(contract: dict, *, speed: float = 0.6) -> JointMotionConfig:
    return JointMotionConfig(tuple(contract["joint_lower"]), tuple(contract["joint_upper"]), max_velocity_rad_s=speed)


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_contract(path: Path) -> dict:
    return validate_contract(json.loads(path.read_text(encoding="utf-8")))


def require_same_contract(expected: dict, actual: dict) -> None:
    validate_contract(expected)
    validate_contract(actual)
    if contract_digest(expected) != contract_digest(actual):
        raise ValueError("policy server and local dataset/robot contracts differ")


def require_home(q, contract: dict, *, tolerance_rad: float = 0.03) -> None:
    if np.max(np.abs(joint_vector(q) - joint_vector(contract["home_q"]))) > tolerance_rad:
        raise RuntimeError("joint start posture differs from the dataset; use the explicit joint prepare command")
