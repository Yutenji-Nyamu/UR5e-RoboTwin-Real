"""Native JAX SFT orchestration, parameter freeze audit, and restorable checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import gc
import hashlib
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np

from .config import BASE_PARAMS, action_expert_path, make_config
from .dataset import validate_dataset, write_json
from .native import CHECKPOINT_COMPAT, load_native_train
from .telemetry import ResourceMonitor, utc_now


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
    # Keep the development plumbing harness/API compatible with older Namespaces.
    defaults = {
        "schedule_steps": None,
        "num_workers": 0,
        "save_interval": 500,
        "eval_interval": 500,
        "eval_points": 3,
        "image_augmentation": False,
        "monitor_interval": 2.0,
        "monitor_console_interval": 30.0,
        "log_base": Path("logs/pi05"),
        "benchmark": False,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if args.eval_interval < 0 or not 1 <= args.eval_points <= 20:
        raise ValueError("evaluation interval must be >=0 and eval_points in 1..20")
    if args.benchmark and (args.resume or not 2 <= args.steps <= 5):
        raise ValueError("benchmark is a fresh 2..5-step probe; no resume or long training")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # make_config validates the experiment name before it is used in a path.
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
        schedule_steps=args.schedule_steps,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        image_augmentation=args.image_augmentation,
    )
    log_dir = args.log_base.resolve() / args.exp_name / stamp
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[RUN_DIR] {log_dir}", flush=True)
    with ResourceMonitor(log_dir, interval_s=args.monitor_interval, console_s=args.monitor_console_interval) as monitor:
        try:
            result = _run_training(args, config, log_dir, monitor)
            write_json(log_dir / "result.json", result)
            return result
        except BaseException as exc:
            monitor.phase("failed")
            write_json(log_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise


def training_recipe(args, config, contract, ready):
    # The LR schedule is immutable, but a later invocation may request more steps
    # within that schedule. This preserves optimizer/step/LR on bounded extensions.
    return {
        "recipe_version": 2,
        "contract": contract,
        "dataset_ready": ready,
        "base_params": str(args.params),
        "native_checkpoint_compat": CHECKPOINT_COMPAT,
        "schedule_steps": config.lr_schedule.decay_steps,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "seed": config.seed,
        "image_augmentation": args.image_augmentation,
        "num_workers": args.num_workers,
        "freeze": "image+language frozen; action expert and action/time projections trainable",
        "ema": None,
    }


def _run_training(args, config, log_dir, monitor):
    logging.basicConfig(level=logging.INFO)
    monitor.phase("load_runtime")
    contract, dataset_ready = validate_dataset(args.dataset)
    native = load_native_train()
    import jax
    from openpi.training import checkpoints, data_loader, sharding

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("native pi05 SFT requires a visible GPU; CPU-only execution is not a valid smoke test")
    recipe = training_recipe(args, config, contract, dataset_ready)
    write_json(log_dir / "recipe.json", recipe)
    write_json(
        log_dir / "invocation.json",
        {
            "requested_steps": args.steps,
            "benchmark": args.benchmark,
            "save_interval": args.save_interval,
            "eval_interval": args.eval_interval,
            "eval_points": args.eval_points,
            "resume": args.resume,
        },
    )
    recipe_path = config.checkpoint_dir.parent / f"{args.exp_name}.recipe.json"
    baseline_path = config.checkpoint_dir.parent / f"{args.exp_name}.freeze_baseline.json"
    manager, resuming = None, False
    if args.benchmark:
        pass  # A probe writes logs only, never a serving checkpoint.
    elif args.resume:
        if json.loads(recipe_path.read_text()) != recipe:
            raise ValueError(
                "resume recipe/dataset changed; preserve batch, schedule, augmentation and optimizer recipe"
            )
    else:
        if config.checkpoint_dir.exists() or recipe_path.exists():
            raise FileExistsError("experiment exists; resume the identical recipe or choose a new experiment name")
        write_json(recipe_path, recipe)
    if not args.benchmark:
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=args.resume
        )
    started = time.monotonic()
    try:
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        monitor.phase("load_dataset")
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        batches = iter(loader)
        data_started = time.monotonic()
        batch = next(batches)
        data_wait_s = time.monotonic() - data_started
        train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
        monitor.phase("load_weights_and_initialize")
        state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=resuming)
        if resuming:
            state = checkpoints.restore_state(manager, state, loader)
        jax.block_until_ready(state)
        monitor.phase("initial_parameter_audit")
        before = parameter_audit(state.params)
        if not args.benchmark:
            if baseline_path.exists():
                if json.loads(baseline_path.read_text())["frozen"] != before["frozen"]:
                    raise RuntimeError("resumed frozen weights differ from the experiment baseline")
            else:
                write_json(baseline_path, before)
        print("[FREEZE] " + json.dumps(before), flush=True)
        step_fn = jax.jit(
            functools.partial(native.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
        last_metrics, step_records, evaluator = {}, [], None
        start_step = int(state.step)
        if start_step >= args.steps:
            raise ValueError("experiment has already reached its requested step count")
        schedule = config.lr_schedule.create()
        with (log_dir / "metrics.jsonl").open("x", buffering=1) as metrics_log:
            for step in range(start_step, args.steps):
                first_step = step == start_step
                monitor.phase("compile_and_first_step" if first_step else "train", step)
                step_started = time.monotonic()
                with sharding.set_mesh(mesh):
                    state, info = step_fn(train_rng, state, batch)
                jax.block_until_ready((state, info))
                compute_s = time.monotonic() - step_started
                last_metrics = {key: float(value) for key, value in jax.device_get(info).items()}
                if not all(np.isfinite(value) for value in last_metrics.values()):
                    raise FloatingPointError(f"nonfinite training metrics: {last_metrics}")
                record = {
                    "time_utc": utc_now(),
                    "step": step + 1,
                    "first_step_in_process": first_step,
                    "compute_s": compute_s,
                    "data_wait_s": data_wait_s,
                    "samples_per_s": args.batch_size / (compute_s + data_wait_s),
                    "learning_rate": float(schedule(step)),
                    "batch_size": args.batch_size,
                    "jax_memory": monitor.jax_stats(jax.devices()),
                    **last_metrics,
                }
                step_records.append(record)
                metrics_log.write(json.dumps(record, allow_nan=False) + "\n")
                if args.benchmark or step % config.log_interval == 0 or step + 1 == args.steps:
                    print(f"[TRAIN] {json.dumps(record)}", flush=True)
                if manager is not None and ((step + 1) % config.save_interval == 0 or step + 1 == args.steps):
                    monitor.phase("save_checkpoint", step + 1)
                    save_started = time.monotonic()
                    checkpoints.save_state(manager, state, loader, step + 1)
                    manager.wait_until_finished()
                    manifest = {
                        "step": step + 1,
                        "contract": contract,
                        "dataset_ready": dataset_ready,
                        "native_checkpoint_compat": CHECKPOINT_COMPAT,
                        "checkpoint_saved": True,
                        "image_augmentation": args.image_augmentation,
                        "status": "intermediate_offline_only",
                        "physical_execution": False,
                    }
                    write_json(config.checkpoint_dir / str(step + 1) / "ur5e_evaluation_manifest.json", manifest)
                    print(f"[SAVE] step={step + 1} elapsed_s={time.monotonic() - save_started:.3f}", flush=True)
                if (
                    not args.benchmark
                    and args.eval_interval
                    and ((step + 1) % args.eval_interval == 0 or step + 1 == args.steps)
                ):
                    monitor.phase("evaluate_training_state", step + 1)
                    eval_started = time.monotonic()
                    if evaluator is None:
                        from .evaluate import StateEvaluator

                        evaluator = StateEvaluator(config, state.model_def, points_per_episode=args.eval_points)
                    evaluation = {"step": step + 1, **evaluator(state.params)}
                    evaluation["elapsed_s"] = time.monotonic() - eval_started
                    write_json(log_dir / f"evaluation_{step + 1}.json", evaluation)
                    print("[EVAL] " + json.dumps({k: v for k, v in evaluation.items() if k != "rows"}), flush=True)
                if step + 1 < args.steps:
                    monitor.phase("load_batch", step + 1)
                    data_started = time.monotonic()
                    batch = next(batches)
                    data_wait_s = time.monotonic() - data_started
        monitor.phase("final_parameter_audit", args.steps)
        after = parameter_audit(state.params)
        if before["frozen"] != after["frozen"] or before["trainable"]["sha256"] == after["trainable"]["sha256"]:
            raise RuntimeError("parameter audit failed: frozen changed or trainable did not change")
        optimizer_sha = array_tree_digest(state.opt_state)
        if manager is not None:
            monitor.phase("release_training_state_before_reload", args.steps)
            # Restore against shapes after releasing training buffers, not beside a second full state.
            template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), state)
            del state
            gc.collect()
            monitor.phase("verify_checkpoint_reload", args.steps)
            restored = checkpoints.restore_state(manager, template, loader, step=args.steps)
            if int(restored.step) != args.steps or parameter_audit(restored.params) != after:
                raise RuntimeError("native checkpoint parameter/step round trip failed")
            if array_tree_digest(restored.opt_state) != optimizer_sha:
                raise RuntimeError("native optimizer checkpoint round trip failed")
        warm = step_records[1:]  # First step includes JIT and is not a steady-state measurement.
        timings = [row["compute_s"] + row["data_wait_s"] for row in warm]
        result = {
            "step": args.steps,
            "metrics": last_metrics,
            "before": before,
            "after": after,
            "frozen_unchanged": True,
            "trainable_changed": True,
            "checkpoint_reload": manager is not None,
            "optimizer_reload": manager is not None,
            "optimizer_sha256": optimizer_sha,
            "elapsed_s": time.monotonic() - started,
            "status": "benchmark_only"
            if args.benchmark
            else ("development_smoke_only" if args.steps < 100 else "SFT_not_physical_validation"),
            "log_dir": str(log_dir),
            "batch_size": args.batch_size,
            "physical_execution": False,
            "measured_warm_steps": len(warm),
            "first_step_compile_and_compute_s": step_records[0]["compute_s"],
            "warm_step_median_s": statistics.median(timings) if timings else None,
            "warm_samples_per_s": args.batch_size * len(warm) / sum(timings) if timings else None,
            "measurement_scope": "few_step_coarse_probe_not_convergence" if args.benchmark else "training_invocation",
        }
        if manager is not None:
            checkpoint = config.checkpoint_dir / str(args.steps)
            write_json(checkpoint / "ur5e_contract.json", contract)
            write_json(checkpoint / "ur5e_verification.json", result)
            write_json(checkpoint / "ur5e_recipe.json", recipe)
            result["checkpoint"] = str(checkpoint)
        monitor.phase("completed", args.steps)
        print("[RESULT] " + json.dumps(result, indent=2), flush=True)
        return result
    finally:
        if manager is not None:
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
    parser.add_argument(
        "--schedule-steps", type=int, help="immutable LR schedule; may exceed this invocation's --steps"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument(
        "--eval-interval", type=int, default=500, help="0 disables periodic/final training-set evaluation"
    )
    parser.add_argument("--eval-points", type=int, default=3, help="fixed, uniformly spaced observations per episode")
    parser.add_argument("--image-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--monitor-console-interval", type=float, default=30.0)
    parser.add_argument("--log-base", type=Path, default=Path("logs/pi05"))
    parser.add_argument("--benchmark", action="store_true", help="2..5 optimizer steps, no checkpoint or evaluation")
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
