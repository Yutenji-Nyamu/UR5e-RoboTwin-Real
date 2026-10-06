"""Training entry point.

    torchrun --nproc_per_node 8 -m metiswam4d.train.train --config configs/stage1_pretrain_human.yaml
    python -m metiswam4d.train.train --config configs/smoke_tiny.yaml --set training.max_steps=5

Single process (no torchrun) runs without FSDP.  With ``WORLD_SIZE > 1`` the
experts are sharded with FSDP2 (``fully_shard``) per Wan block plus the root,
BF16 parameters / FP32 reduction, and checkpoints are written with DCP.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from metiswam4d.attention import dense_to_compact_bias
from metiswam4d.build import build_model, build_schedule, initialize_model
from metiswam4d.config import StageConfig, config_to_dict, load_stage_config
from metiswam4d.data.contract import move_batch
from metiswam4d.objectives import compute_losses, prepare_training_step
from metiswam4d.train.checkpoint import (
    is_main, latest_checkpoint, load_checkpoint, save_checkpoint,
)
from metiswam4d.train.curriculum import Curriculum
from metiswam4d.train.data import build_loader, build_val_loaders
from metiswam4d.train.visualize import build_visualizer


def _parse_override(item: str) -> tuple[str, Any]:
    key, _, value = item.partition("=")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def setup_distributed() -> tuple[int, int, torch.device]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world, device


def start_gpu_filler(device: torch.device):
    """Queue monitors average ``utilization.gpu``; a training step leaves gaps (collectives, host work, checkpoint
    writes) that a second CUDA process could only fill by time-slicing the training context.  Sleep kernels on a
    lowest-priority stream of the training context itself run next to the training kernels on one SM instead.
    Returns a stop function; the thread must end before interpreter shutdown (a live CUDA thread aborts the exit)."""
    import threading
    stop = threading.Event()

    def fill() -> None:
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device, priority=torch.cuda.Stream.priority_range()[0])
        while not stop.is_set():
            with torch.cuda.stream(stream):
                torch.cuda._sleep(20_000_000)
            stream.synchronize()

    thread = threading.Thread(target=fill, name="gpu-filler", daemon=True)
    thread.start()

    def stop_filler() -> None:
        stop.set()
        thread.join()
    return stop_filler


def wrap_fsdp(model: nn.Module, dtype: torch.dtype, granularity: str = "block", *, hybrid: bool = True,
              reshard_after_forward: bool = False, reduce_dtype: torch.dtype = torch.bfloat16,
              world: int = 1, local_world: int = 1) -> nn.Module:
    """FSDP2.  ``block``: every Wan block is its own shard unit (parameters are gathered per layer
    because the block halves run through ``WanStyleBlock.forward``); ``root``: one unit for the whole
    model.  ``hybrid`` shards inside a node and replicates across nodes so the cross-node traffic is
    one gradient reduction per step instead of repeated weight all-gathers."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from metiswam4d.experts.local import WanStyleBlock
    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=reduce_dtype)
    nodes = world // local_world
    if hybrid and nodes > 1 and world % local_world == 0:
        mesh = init_device_mesh("cuda", (nodes, local_world), mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))
    kwargs = dict(mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    if granularity == "block":
        for module in model.modules():
            if isinstance(module, WanStyleBlock):
                fully_shard(module, **kwargs)
    elif granularity != "root":
        raise ValueError(f"unknown fsdp granularity {granularity!r}")
    fully_shard(model, **kwargs)
    return model


def focus_diagnostics(model: nn.Module) -> dict[str, float]:
    """Scalars of the reading interface: saliency bias strength per modality, the norm of the compact-read
    output projections (how much the Action can receive), and the Track feedback gate.  The actual read
    magnitude per step is ``focus/read_ratio`` from the losses."""
    values: dict[str, float] = {}
    reader = getattr(model, "reader", None)
    if reader is None:
        return values

    def scalar(param: torch.Tensor) -> torch.Tensor:
        t = param.detach()
        if hasattr(t, "full_tensor"):  # FSDP2 DTensor
            t = t.full_tensor()
        return t.float().reshape(-1)[0].cpu()

    with torch.no_grad():
        for name, csia in reader.csia.items():
            values[f"focus/gamma_{name}"] = float(torch.nn.functional.softplus(scalar(csia.gamma_raw)))
        norms = []
        for m in getattr(model, "focus_attention", {}).values():
            w = m.o.weight.detach()
            if hasattr(w, "full_tensor"):
                w = w.full_tensor()
            norms.append(float(w.float().norm()))
        if norms:
            values["focus/read_out_norm_mean"] = sum(norms) / len(norms)
        if reader.feedback is not None:
            w = reader.feedback.attn.out_proj.weight.detach()
            if hasattr(w, "full_tensor"):
                w = w.full_tensor()
            values["focus/track_feedback_out_norm"] = float(w.float().norm())
    return values


class JsonlLogger:
    def __init__(self, path: Path, enabled: bool):
        self.enabled = enabled
        self.handle = open(path, "a") if enabled else None
        self.tb = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(path.parent / "tensorboard"))
            except Exception:  # pragma: no cover - tensorboard optional
                self.tb = None

    def log(self, step: int, values: dict[str, float]) -> None:
        if not self.enabled:
            return
        record = {"step": step, "time": time.time(), **values}
        self.handle.write(json.dumps(record) + "\n")
        self.handle.flush()
        if self.tb is not None:
            for key, value in values.items():
                self.tb.add_scalar(key, value, step)

    def close(self) -> None:
        if self.handle:
            self.handle.close()
        if self.tb is not None:
            self.tb.close()


def start_stall_watchdog(rank: int, timeout_s: float = 480.0):
    """When no step completes for ``timeout_s`` (below the 600 s NCCL collective timeout), dump the Python
    stacks of this rank and of its DataLoader workers to stderr.  Returns the heartbeat callable."""
    import faulthandler
    import multiprocessing
    import signal
    import threading

    faulthandler.register(signal.SIGUSR1, all_threads=True)  # inherited by DataLoader workers forked later
    last = [time.time()]

    def watch() -> None:
        fired = False
        while True:
            time.sleep(15)
            idle = time.time() - last[0]
            if idle < timeout_s:
                fired = False
                continue
            if fired:
                continue
            fired = True
            workers = multiprocessing.active_children()
            print(f"[stall] rank {rank}: no step for {idle:.0f}s; stacks of the rank and {len(workers)} workers "
                  f"{[w.pid for w in workers]} follow", file=sys.stderr, flush=True)
            faulthandler.dump_traceback(all_threads=True)
            for w in workers:  # one at a time: they share stderr
                os.kill(w.pid, signal.SIGUSR1)
                time.sleep(1)

    threading.Thread(target=watch, name="stall-watchdog", daemon=True).start()
    return lambda: last.__setitem__(0, time.time())


def train(stage: StageConfig, *, stop_after_steps: int | None = None) -> dict[str, float]:
    rank, world, device = setup_distributed()
    tcfg = stage.training
    torch.manual_seed(tcfg.seed + rank)
    random.seed(tcfg.seed + rank)
    out_dir = Path(stage.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.resolved.json").write_text(json.dumps(config_to_dict(stage), indent=2, default=str))
    log = (lambda *a, **k: print(*a, **k, flush=True)) if is_main() else (lambda *a, **k: None)
    stop_filler = None
    if tcfg.gpu_filler and device.type == "cuda":
        stop_filler = start_gpu_filler(device)
        log("[gpu] utilisation filler on a lowest-priority stream")

    param_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[tcfg.dtype]
    model = build_model(stage.model)
    initialize_model(model, stage, log=log)
    counts = {name: sum(p.numel() for p in ps) for name, ps in model.trainable_groups().items()}
    log(f"[model] parameters by group: { {k: f'{v/1e6:.1f}M' for k, v in counts.items()} }")

    use_fsdp = tcfg.fsdp and world > 1
    if use_fsdp:
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() or 1))
        model = wrap_fsdp(
            model, param_dtype, tcfg.fsdp_granularity, hybrid=tcfg.fsdp_hybrid,
            reshard_after_forward=tcfg.fsdp_reshard_after_forward,
            reduce_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[tcfg.reduce_dtype],
            world=world, local_world=local_world)
        model.to(device)
        log(f"[fsdp] world={world} local={local_world} hybrid={tcfg.fsdp_hybrid and world // local_world > 1} "
            f"granularity={tcfg.fsdp_granularity} reshard_after_forward={tcfg.fsdp_reshard_after_forward} "
            f"reduce={tcfg.reduce_dtype} grad_accumulation={tcfg.grad_accumulation}")
    else:
        model.to(device=device, dtype=param_dtype if device.type == "cuda" else torch.float32)
    groups = model.trainable_groups()
    curriculum = Curriculum(groups, tcfg)
    optimizer = curriculum.build_optimizer()
    if curriculum.unused_groups:
        log(f"[curriculum] lr specs without parameters (ignored): {curriculum.unused_groups}")

    step = 0
    ckpt_dir = out_dir / "checkpoints"
    resume = stage.init.resume_from
    if resume == "auto":
        resume = latest_checkpoint(ckpt_dir)
    if resume:
        step = load_checkpoint(Path(resume), model, optimizer, log=log)
        log(f"[resume] from {resume} at step {step}")
    elif stage.init.start_step > 0:
        step = int(stage.init.start_step)
        log(f"[init] curriculum continues from step {step} (weights from initialize_from, fresh optimizer)")
    if is_main():  # the exact sources of every (re)start live next to the checkpoints
        from metiswam4d.train.snapshot import snapshot_code
        project = Path(__file__).resolve().parents[2]
        archive = snapshot_code(project, out_dir, step=step, extra={
            "stage": stage.name, "start_step": step, "resume_from": resume, "world_size": world,
            "torch": torch.__version__, "python": sys.version.split()[0], "argv": " ".join(sys.argv)})
        log(f"[snapshot] code -> {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    schedule, mixture = build_schedule(stage)
    accum = max(1, int(tcfg.grad_accumulation))
    loader, sampler = build_loader(stage.data, num_batches=max(1, tcfg.max_steps - step) * accum, seed=tcfg.seed,
                                   rank=rank, world_size=world, epoch=step)
    encoder = None
    val_loaders: dict = {}
    if stage.data.rt2 is not None:
        from metiswam4d.data.rt2 import RT2OnlineEncoder
        encoder = RT2OnlineEncoder(stage.data.rt2_encoder, device)
        log(f"[data] RT2 episodes={len(loader.dataset) // stage.data.rt2.samples_per_episode} "
            f"windows/epoch={len(loader.dataset)} batches/rank/epoch={len(loader)} online Wan-VAE encoding")
    elif stage.data.robodojo is not None:
        from metiswam4d.data.rt2 import RT2OnlineEncoder
        encoder = RT2OnlineEncoder(stage.data.robodojo_encoder, device)
        log(f"[data] RoboDojo {stage.data.robodojo.split} episodes={len(loader.dataset) // stage.data.robodojo.samples_per_episode} "
            f"windows/epoch={len(loader.dataset)} batches/rank/epoch={len(loader)} online Wan-VAE encoding")
        val_loaders = build_val_loaders(stage.data, seed=tcfg.seed, rank=rank, world_size=world)
        for name, vl in val_loaders.items():
            log(f"[data] {name}: {len(vl.dataset.rows)} held-out RoboDojo episodes, {len(vl)} batches/rank")
    elif stage.data.human is not None:
        from metiswam4d.data.human import HumanEncoderConfig, HumanOnlineEncoder
        encoder = HumanOnlineEncoder(stage.data.human_encoder or HumanEncoderConfig(), device)
        sizes = {c.name: len(c.dataset) for c in loader.dataset.components}
        log(f"[data] human mixture {sizes} weights={ {c.name: c.weight for c in loader.dataset.components} } "
            f"online Wan-VAE + UMT5 encoding")
        val_loaders = build_val_loaders(stage.data, seed=tcfg.seed, rank=rank, world_size=world)
        for name, vl in val_loaders.items():
            log(f"[data] {name}: {len(vl.dataset.rows)} held-out IG-10K {vl.dataset.config.profile} episodes, "
                f"{len(vl)} batches/rank")
    logger = JsonlLogger(out_dir / "train_log.jsonl", is_main())
    generator = torch.Generator().manual_seed(tcfg.seed * 31 + rank + step)
    compute_dtype = param_dtype if device.type == "cuda" else torch.float32
    visualizer = build_visualizer(stage, encoder=encoder, device=device, dtype=compute_dtype, rank=rank, world=world,
                                  out_dir=out_dir, is_main=is_main(), tb=logger.tb, schedule=schedule, log=log)

    def checkpoint_due(step: int, last_save: float) -> bool:
        """Step-based, or wall-clock based (rank 0 decides, everyone follows: DCP save is collective)."""
        if tcfg.checkpoint_every_minutes > 0:
            due = (time.time() - last_save) >= tcfg.checkpoint_every_minutes * 60.0
            if dist.is_initialized():
                flag = torch.tensor([int(due)], device=device if device.type == "cuda" else None)
                dist.broadcast(flag, src=0)
                due = bool(flag.item())
            return due
        return tcfg.checkpoint_every > 0 and step % tcfg.checkpoint_every == 0

    def checkpoint_and_visualize(step: int, final: bool = False) -> None:
        save_checkpoint(ckpt_dir, model, optimizer, step=step, keep_last=tcfg.keep_last, extra={"stage": stage.name})
        log(f"[checkpoint] {'final ' if final else ''}step {step}")
        if visualizer is not None:
            t0 = time.time()
            scalars = visualizer.run(model, step)
            logger.log(step, scalars)
            log(f"[vis] step {step} " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in scalars.items()) +
                f" ({time.time() - t0:.0f}s)")

    @torch.no_grad()
    def evaluate(step: int) -> None:
        """Losses on the held-out batches of every split with a fixed noise draw; every rank runs the same batch
        count per split (the splits differ in modality presence, so each is a separate pass)."""
        if not val_loaders:
            return
        model.eval()
        for name, val_loader in val_loaders.items():
            t0 = time.time()
            gen = torch.Generator().manual_seed(tcfg.seed * 977 + 1)
            sums: dict[str, float] = {}
            count = 0
            for raw in val_loader:
                batch = encoder(move_batch(raw, device)) if encoder is not None else raw
                batch = move_batch(batch, device, compute_dtype)
                ts = prepare_training_step(batch, schedule=schedule, mixture=mixture, dropout=None, generator=gen,
                                           dtype=compute_dtype)
                with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=device.type == "cuda"):
                    out = model(ts.inputs)
                losses = compute_losses(out, ts, weights=tcfg.loss, track_regions=tcfg.track_regions)
                for k, v in losses.items():
                    if k.startswith("loss/"):
                        sums[k] = sums.get(k, 0.0) + float(v)
                count += 1
            values = {f"{name}/{k.split('/', 1)[1]}": v / max(count, 1) for k, v in sums.items()}
            if dist.is_initialized():
                keys = sorted(values)
                t = torch.tensor([values[k] for k in keys], device=device if device.type == "cuda" else None)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                values = {k: float(t[i]) / dist.get_world_size() for i, k in enumerate(keys)}
            logger.log(step, values)
            log(f"[eval] step {step} {name}: " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in values.items()) +
                f" ({time.time() - t0:.0f}s)")
        model.train()

    last_save = time.time()
    last_saved_step = step

    model.train()
    last: dict[str, float] = {}
    t_last = time.time()
    target_steps = tcfg.max_steps if stop_after_steps is None else min(tcfg.max_steps, step + stop_after_steps)
    epoch = step
    heartbeat = start_stall_watchdog(rank)
    loader_iter = iter(loader)

    def next_batch():
        nonlocal loader_iter, epoch
        try:
            return next(loader_iter)
        except StopIteration:
            if encoder is None:  # finite mixture schedule
                return None
            epoch += 1
            sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            return next(loader_iter)

    phase_time = {"data": 0.0, "encode": 0.0, "fwd_bwd": 0.0, "optim": 0.0}
    window: dict[str, list[float]] = {}   # loss key -> [sum, count] over the steps since the last log line

    def batch_kind(b: dict) -> str:
        if b.get("track_clean") is None:
            return "video_only"
        if b.get("action") is not None:
            return "track_action"
        return "track_with_object" if bool((b["track_role"] == 2).any()) else "track_body_only"

    while step < target_steps:
        lrs = curriculum.step(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        exhausted = False
        for micro in range(accum):
            t_phase = time.time()
            batch = next_batch()
            if batch is None:
                exhausted = True
                break
            phase_time["data"] += time.time() - t_phase
            t_phase = time.time()
            if encoder is not None:
                batch = encoder(move_batch(batch, device))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            phase_time["encode"] += time.time() - t_phase
            t_phase = time.time()
            batch = move_batch(batch, device, compute_dtype)
            kind = window.setdefault(f"batches/{batch_kind(batch)}", [0.0, 0])
            kind[0] += 1.0
            training_step = prepare_training_step(
                batch, schedule=schedule, mixture=mixture, dropout=tcfg.dropout, generator=generator, dtype=compute_dtype)
            if stage.model.action_read == "compact" and tcfg.dense_to_compact_steps > 0:
                training_step.inputs.dense_read_bias = dense_to_compact_bias(
                    step, tcfg.dense_to_compact_steps, tcfg.dense_to_compact_max_bias)
            if use_fsdp and accum > 1:
                model.set_requires_gradient_sync(micro == accum - 1)
            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=device.type == "cuda"):
                output = model(training_step.inputs)
            losses = compute_losses(output, training_step, weights=tcfg.loss, track_regions=tcfg.track_regions)
            (losses["total"] / accum).backward()
            for k, v in losses.items():
                if k != "total":
                    accumulated[k] = accumulated.get(k, 0.0) + float(v) / accum
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phase_time["fwd_bwd"] += time.time() - t_phase
        if exhausted:
            break
        losses = accumulated
        for k, v in losses.items():  # batches differ in modality; the log line reports window means per key
            entry = window.setdefault(k, [0.0, 0])
            entry[0] += v
            entry[1] += 1
        t_phase = time.time()
        if tcfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        else:
            grad_norm = torch.zeros(())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_time["optim"] += time.time() - t_phase
        step += 1
        heartbeat()

        if step % tcfg.log_every == 0 or step == target_steps:
            now = time.time()
            values = {k: (s / n if n else s) for k, (s, n) in window.items()}  # batches/* are counts (n = 0)
            window.clear()
            values.update(lrs)
            values["grad_norm"] = float(grad_norm)
            if stage.model.action_read == "compact" and tcfg.dense_to_compact_steps > 0:
                bias = dense_to_compact_bias(step, tcfg.dense_to_compact_steps, tcfg.dense_to_compact_max_bias)
                values["anneal/dense_bias"] = float("-inf") if bias is None else bias
            values.update(focus_diagnostics(model))
            values["step_time"] = (now - t_last) / tcfg.log_every
            for name, seconds in phase_time.items():
                values[f"time/{name}"] = seconds / tcfg.log_every
                phase_time[name] = 0.0
            if device.type == "cuda":
                values["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            t_last = now
            logger.log(step, values)
            log(f"[step {step}] loss={values['loss/total']:.4f} " +
                " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in values.items() if k.startswith("loss/") and k != "loss/total") +
                f" | frozen={curriculum.frozen_groups(step)} | {values['step_time']:.2f}s/it "
                f"(data {values['time/data']:.1f} enc {values['time/encode']:.1f} fb {values['time/fwd_bwd']:.1f} opt {values['time/optim']:.1f})")
            last = values
        if tcfg.eval_every > 0 and step % tcfg.eval_every == 0:
            evaluate(step)
            heartbeat()
        if checkpoint_due(step, last_save):
            checkpoint_and_visualize(step)
            last_save = time.time()
            last_saved_step = step
            heartbeat()
    if (tcfg.checkpoint_every > 0 or tcfg.checkpoint_every_minutes > 0) and step != last_saved_step:
        checkpoint_and_visualize(step, final=True)
    logger.close()
    if stop_filler is not None:
        stop_filler()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return last


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MetisWAM4D stage trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. training.max_steps=10")
    parser.add_argument("--stop-after-steps", type=int, default=None)
    args = parser.parse_args(argv)
    overrides = dict(_parse_override(item) for item in args.set)
    stage = load_stage_config(args.config, overrides)
    train(stage, stop_after_steps=args.stop_after_steps)


if __name__ == "__main__":
    main(sys.argv[1:])
