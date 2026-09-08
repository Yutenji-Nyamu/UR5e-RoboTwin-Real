"""Native JAX SFT orchestration, parameter freeze audit, and restorable checkpoints."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np

from .config import BASE_PARAMS, action_expert_path, make_config
from .dataset import validate_dataset, write_json
from .native import CHECKPOINT_COMPAT, load_native_train


def parameter_audit(params):
    """Hash one leaf at a time on the host; never keep a second full model on GPU."""
    import flax.traverse_util
    from flax import nnx
    import jax

    groups = {key: {"elements": 0, "leaves": 0, "digest": hashlib.sha256()} for key in ("frozen", "trainable")}
    for path, value in sorted(flax.traverse_util.flatten_dict(params.filter(nnx.Param).to_pure_dict()).items()):
        array = np.asarray(jax.device_get(value))
        group = groups["trainable" if action_expert_path(path) else "frozen"]
        group["elements"] += array.size
        group["leaves"] += 1
        group["digest"].update(("/".join(path) + str(array.dtype) + str(array.shape)).encode())
        group["digest"].update(array.tobytes())
    result = {
        key: {"elements": value["elements"], "leaves": value["leaves"], "sha256": value["digest"].hexdigest()}
        for key, value in groups.items()
    }
    if result["trainable"]["elements"] == 0 or result["frozen"]["elements"] == 0:
        raise RuntimeError("freeze filter did not select both action expert and frozen backbones")
    return result


def array_tree_digest(tree):
    import jax

    digest = hashlib.sha256()
    for value in jax.tree.leaves(tree):
        array = np.asarray(jax.device_get(value))
        digest.update((str(array.dtype) + str(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_training(args):
    logging.basicConfig(level=logging.INFO)
    contract, dataset_ready = validate_dataset(args.dataset)
    config = make_config(
        args.dataset,
        exp_name=args.exp_name,
        checkpoint_base=args.checkpoint_base,
        params=args.params,
        steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )
    native = load_native_train()
    import jax
    from openpi.training import checkpoints, data_loader, sharding

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("native pi05 SFT requires a visible GPU; CPU-only execution is not a valid smoke test")
    recipe = {
        "contract": contract,
        "dataset_ready": dataset_ready,
        "base_params": str(args.params),
        "native_checkpoint_compat": CHECKPOINT_COMPAT,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "seed": config.seed,
        "freeze": "image+language frozen; action expert and action/time projections trainable",
        "ema": None,
    }
    recipe_path = config.checkpoint_dir.parent / f"{args.exp_name}.recipe.json"
    if args.resume:
        if json.loads(recipe_path.read_text()) != recipe:
            raise ValueError("resume recipe/dataset changed; use a new experiment instead")
    else:
        if config.checkpoint_dir.exists() or recipe_path.exists():
            raise FileExistsError("experiment exists; resume the identical recipe or choose a new experiment name")
        write_json(recipe_path, recipe)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=args.resume
    )
    started = time.monotonic()
    try:
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        batches = iter(loader)
        batch = next(batches)
        train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
        state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=resuming)
        if resuming:
            state = checkpoints.restore_state(manager, state, loader)
        jax.block_until_ready(state)
        before = parameter_audit(state.params)
        print("[FREEZE] " + json.dumps(before), flush=True)
        step_fn = jax.jit(
            functools.partial(native.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
        last_metrics = {}
        start_step = int(state.step)
        if start_step >= args.steps:
            raise ValueError("experiment has already reached its requested step count")
        for step in range(start_step, args.steps):
            with sharding.set_mesh(mesh):
                state, info = step_fn(train_rng, state, batch)
            last_metrics = {key: float(value) for key, value in jax.device_get(info).items()}
            if not all(np.isfinite(value) for value in last_metrics.values()):
                raise FloatingPointError(f"nonfinite training metrics: {last_metrics}")
            if step % config.log_interval == 0 or step + 1 == args.steps:
                print(f"[TRAIN] step={step + 1} {last_metrics}", flush=True)
            if (step + 1) % config.save_interval == 0 or step + 1 == args.steps:
                checkpoints.save_state(manager, state, loader, step + 1)
            if step + 1 < args.steps:
                batch = next(batches)
        manager.wait_until_finished()
        after = parameter_audit(state.params)
        if before["frozen"] != after["frozen"] or before["trainable"]["sha256"] == after["trainable"]["sha256"]:
            raise RuntimeError("parameter audit failed: frozen changed or trainable did not change")
        # Restore full training state (parameters + optimizer), not just existence of files.
        restored = checkpoints.restore_state(manager, state, loader, step=args.steps)
        if int(restored.step) != args.steps or parameter_audit(restored.params) != after:
            raise RuntimeError("native checkpoint parameter/step round trip failed")
        optimizer_sha = array_tree_digest(state.opt_state)
        if array_tree_digest(restored.opt_state) != optimizer_sha:
            raise RuntimeError("native optimizer checkpoint round trip failed")
        result = {
            "step": args.steps,
            "metrics": last_metrics,
            "before": before,
            "after": after,
            "frozen_unchanged": True,
            "trainable_changed": True,
            "checkpoint_reload": True,
            "optimizer_reload": True,
            "optimizer_sha256": optimizer_sha,
            "elapsed_s": time.monotonic() - started,
            "status": "development_smoke_only" if args.steps < 100 else "SFT_not_physical_validation",
        }
        checkpoint = config.checkpoint_dir / str(args.steps)
        write_json(checkpoint / "ur5e_contract.json", contract)
        write_json(checkpoint / "ur5e_verification.json", result)
        write_json(checkpoint / "ur5e_recipe.json", recipe)
        print(f"[CHECKPOINT] {checkpoint}\n" + json.dumps(result, indent=2), flush=True)
    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-base", type=Path, default=Path("checkpoints/pi05"))
    parser.add_argument("--params", default=BASE_PARAMS)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--resume", action="store_true")
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
