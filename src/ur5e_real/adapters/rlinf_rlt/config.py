"""Simulation algorithm parameters and real-world scheduling are separate sections."""

from __future__ import annotations

from copy import deepcopy
import math

DONOR_COMMIT = "d3acd650869d376c14c1406d90865ef90d680432"
FEATURE_VERSION = "ur5e_pi05_last_prefix_v1"
ACTION_VERSION = "ur5e_rlt_affine_joint7_v1"

DEFAULTS = {
    "schema": 1,
    "donor_commit": DONOR_COMMIT,
    "token": {
        "input_dim": 2048,
        "embed_dim": 2048,
        "prefix_seq_len": 768,
        "num_layers": 2,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.0,
    },
    "token_train": {
        "steps": 500,
        "diagnostic_interval": 100,
        "save_interval": 100,
        "batch_size": 2,
        "learning_rate": 2.5e-5,
        "warmup_steps": 100,
        "weight_decay": 1e-10,
        "seed": 0,
    },
    "algorithm": {
        "hidden_dim": 256,
        "hidden_layers": 3,
        "proprio_dim": 13,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "gamma": 0.99,
        "tau": 0.005,
        "critic_actor_ratio": 2,
        "reference_dropout": 0.5,
        "fixed_std": 0.002,
        "warmup_bc_weight": 7.0,
        "warmup_q_weight": 0.05,
        "online_bc_weight": 2.5,
        "online_q_weight": 0.45,
        "ramp_updates": 10000,
        "gradient_clip": 10.0,
        "updates_per_transition": 2,
        "bootstrap_on_timeout": True,
    },
    "real": {
        "episodes_per_round": 4,
        "action_steps": 20,
        "max_chunks": 30,
        "speed_rad_s": 0.6,
        "rpc_timeout_s": 1.5,
        "release_hold_s": 1.0,
        "success_instruction": "确认方块已抓起、放下并完全释放；按现场任务标准标注成功或失败。",
    },
    "save": {"every_update": True, "keep_last": 20},
}


def configuration(overrides=None):
    result = deepcopy(DEFAULTS)
    for key, value in (overrides or {}).items():
        if key not in result:
            raise ValueError(f"unknown RLT configuration section: {key}")
        if isinstance(result[key], dict):
            unknown = set(value) - set(result[key])
            if unknown:
                raise ValueError(f"unknown {key} settings: {sorted(unknown)}")
            result[key].update(value)
        else:
            result[key] = value
    if result["schema"] != 1 or result["donor_commit"] != DONOR_COMMIT:
        raise ValueError("RLT schema/donor lock mismatch")
    real, token, train, algorithm = (result[name] for name in ("real", "token", "token_train", "algorithm"))
    for name in ("real", "token", "token_train", "algorithm", "save"):
        for key, value in result[name].items():
            default = DEFAULTS[name][key]
            if type(default) is bool and type(value) is not bool:
                raise ValueError(f"{name}.{key} must be boolean")
            if type(default) is int and (
                type(value) is not int or value < (0 if key in ("seed", "warmup_steps") else 1)
            ):
                raise ValueError(f"{name}.{key} must be a valid integer")
            if isinstance(default, float) and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{name}.{key} must be a finite number")
    if result["save"]["every_update"] is not True:
        raise ValueError("this implementation saves paired A/C state after every learner update")
    if min(token["mlp_ratio"], train["learning_rate"], algorithm["learning_rate"], algorithm["gradient_clip"]) <= 0:
        raise ValueError("width ratio, learning rates and clipping norm must be positive")
    if not 0 <= token["dropout_rate"] < 1 or train["weight_decay"] < 0:
        raise ValueError("invalid token dropout or weight decay")
    if any(
        algorithm[key] < 0 for key in ("warmup_bc_weight", "warmup_q_weight", "online_bc_weight", "online_q_weight")
    ):
        raise ValueError("loss weights must be nonnegative")
    if real["episodes_per_round"] != 4 or not 1 <= real["action_steps"] <= 50 or real["max_chunks"] < 1:
        raise ValueError("use four sequential episodes per round, K in 1..50, and positive max_chunks")
    if not 0 < real["speed_rad_s"] <= 0.6 or not 0 < real["rpc_timeout_s"] <= 1.5:
        raise ValueError("real-time limits must match the existing joint executor")
    if real["release_hold_s"] < 1 or not real["success_instruction"].strip():
        raise ValueError("retain at least one second after release and an explicit human result instruction")
    if (
        token["embed_dim"] % token["num_heads"]
        or min(token[k] for k in ("input_dim", "embed_dim", "prefix_seq_len", "num_layers", "num_heads")) < 1
    ):
        raise ValueError("invalid token dimensions")
    if algorithm["proprio_dim"] != 13 or algorithm["critic_actor_ratio"] != 2:
        raise ValueError("UR5 state is 7 native state values + 6 executor values; critic:actor must be 2:1")
    for section in (real, token, train, algorithm):
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in section.values()):
            raise ValueError("non-finite RLT parameter")
    if not 0 < algorithm["gamma"] <= 1 or not 0 < algorithm["tau"] <= 1 or not 0 <= algorithm["reference_dropout"] < 1:
        raise ValueError("invalid discount, target update, or reference dropout")
    if any(train[k] < 1 for k in ("steps", "batch_size", "diagnostic_interval", "save_interval")):
        raise ValueError("training lengths, batches, and intervals must be positive")
    if algorithm["batch_size"] < 1 or algorithm["fixed_std"] <= 0 or result["save"]["keep_last"] < 1:
        raise ValueError("invalid learner batch, exploration standard deviation, or checkpoint retention")
    return result
