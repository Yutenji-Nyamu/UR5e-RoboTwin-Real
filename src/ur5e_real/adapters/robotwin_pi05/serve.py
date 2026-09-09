"""Serve a verified native checkpoint over the original local WebSocket protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .config import make_config
from .contract import STATE_LAYOUT, decode_actions, read_contract, require_same_contract
from .dataset import offline_observation, validate_dataset
from .native import CHECKPOINT_COMPAT


def load_policy(dataset: Path, checkpoint: Path, diffusion_steps=10):
    contract, ready = validate_dataset(dataset)
    require_same_contract(contract, read_contract(checkpoint / "ur5e_contract.json"))
    recipe = json.loads((checkpoint / "ur5e_recipe.json").read_text())
    require_same_contract(contract, recipe["contract"])
    if recipe.get("native_checkpoint_compat") != CHECKPOINT_COMPAT or (checkpoint / "model.safetensors").exists():
        raise ValueError("checkpoint must use the locked native JAX/Orbax runtime")
    verification = json.loads((checkpoint / "ur5e_verification.json").read_text())
    if not all(
        verification.get(key) is True
        for key in ("frozen_unchanged", "trainable_changed", "checkpoint_reload", "optimizer_reload")
    ):
        raise ValueError("checkpoint did not pass the native training/freeze/reload audit")
    norm_file = checkpoint / "assets" / STATE_LAYOUT / "norm_stats.json"
    if hashlib.sha256(norm_file.read_bytes()).hexdigest() != ready["norm_sha256"]:
        raise ValueError("checkpoint and dataset normalization statistics differ")
    if not 1 <= diffusion_steps <= 50:
        raise ValueError("diffusion steps must be in [1, 50]")
    config = make_config(dataset)
    from openpi.policies.policy_config import create_trained_policy

    policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": diffusion_steps})
    return policy, contract, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--port", default=8005, type=int)
    parser.add_argument("--diffusion-steps", default=10, type=int)
    parser.add_argument("--instance-id", help="internal identity for an operator-owned model process")
    args = parser.parse_args()
    policy, contract, verification = load_policy(args.dataset, args.checkpoint, args.diffusion_steps)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    observation, _ = offline_observation(args.dataset)
    for i in range(2):
        started = time.monotonic()
        decode_actions(policy.infer(observation)["actions"])
        print(f"[WARMUP] {i + 1}: {(time.monotonic() - started) * 1000:.1f}ms", flush=True)
    metadata = {
        "ur5e_contract": contract,
        "checkpoint_step": verification["step"],
        "training_status": verification["status"],
        "diffusion_steps": args.diffusion_steps,
        "instance_id": args.instance_id,
        "checkpoint_path": str(args.checkpoint.resolve()),
    }
    print(f"[READY] ws://127.0.0.1:{args.port} ; no robot connection", flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
