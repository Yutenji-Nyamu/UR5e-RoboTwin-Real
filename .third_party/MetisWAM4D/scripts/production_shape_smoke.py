"""One full-size training step on synthetic, contract-shaped data.

Builds the production model (Alpha Video 5B + Alpha Action 1B + interpolated Track), loads the real
Alpha weights, and runs forward + backward for one batch in the requested reading mode on a single
GPU with activation checkpointing.  Reports parameter counts, peak memory and timing.

    source scripts/env.sh
    python scripts/production_shape_smoke.py --alpha <alpha_dir> [--experts video track action] [--action-read compact]
"""
from __future__ import annotations

import argparse
import time

import torch

from metiswam4d.build import build_model, build_schedule, initialize_model
from metiswam4d.config import InitConfig, StageConfig, TrackInit, load_stage_config
from metiswam4d.data import SyntheticLatentDataset, collate
from metiswam4d.data.contract import move_batch
from metiswam4d.objectives import DropoutConfig, LossWeights, compute_losses, prepare_training_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--config", default="configs/stage2_midtrain_ig10k.yaml")
    parser.add_argument("--experts", nargs="+", default=None)
    parser.add_argument("--action-read", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1, help="repeat the step; the first one includes warm-up")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    stage: StageConfig = load_stage_config(args.config)
    if args.experts:
        stage.model.experts = tuple(args.experts)
    if args.action_read:
        stage.model.action_read = args.action_read
    stage.init = InitConfig(alpha_checkpoint=args.alpha, load_action_from_alpha="action" in stage.model.experts,
                            track=TrackInit(mode="interpolate_from_video"))
    if args.no_checkpointing:
        stage.model.gradient_checkpointing = False
    device = torch.device(args.device)

    t0 = time.time()
    model = build_model(stage.model)
    initialize_model(model, stage)
    counts = {k: sum(p.numel() for p in v) for k, v in model.trainable_groups().items()}
    print(f"[build] {time.time() - t0:.0f}s params: { {k: f'{v / 1e9:.3f}B' for k, v in counts.items()} }", flush=True)
    model.to(device=device, dtype=torch.bfloat16).train()

    spec = stage.data.spec
    ds = SyntheticLatentDataset(spec, args.batch, with_action="action" in stage.model.experts, text_len=64)
    batch = move_batch(collate([ds[i] for i in range(args.batch)]), device, torch.bfloat16)
    schedule, mixture = build_schedule(stage)
    step = prepare_training_step(batch, schedule=schedule, mixture=mixture, dropout=DropoutConfig(),
                                 generator=torch.Generator().manual_seed(0), dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(device)
    for i in range(args.steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(step.inputs)
        t_fwd = time.time()
        losses = compute_losses(out, step, weights=LossWeights())
        losses["total"].backward()
        torch.cuda.synchronize(device)
        print(f"[step {i}] {time.time() - t0:.2f}s (forward {t_fwd - t0:.2f}s)  loss={float(losses['total']):.4f} "
              + " ".join(f"{k.split('/')[1]}={float(v):.3f}" for k, v in losses.items() if k.startswith("loss/") and k != "loss/total"),
              flush=True)
    print(f"[memory] peak {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB "
          f"(params+grads bf16 ~{2 * 2 * sum(counts.values()) / 1e9:.1f} GB)")
    shapes = {k: tuple(getattr(out, f"{k}_velocity").shape) for k in stage.model.experts}
    print(f"[shapes] {shapes} focus_layers={len(out.focus)} "
          f"tokens={ {k: tuple(v.shape) for k, v in (out.focus[-1].tokens.items() if out.focus else [])} }")


if __name__ == "__main__":
    main()
