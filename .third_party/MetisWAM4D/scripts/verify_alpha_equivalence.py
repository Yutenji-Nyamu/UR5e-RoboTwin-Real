"""Numerically compare the local Wan-compatible experts against the vendored OpenWAM-Alpha.

Loads the same Alpha checkpoint into (a) the official ``DualSystemSelfAttnArchitecture`` from the
vendored ``openwam`` package and (b) ``MetisWAM4D(video=LatentExpert, action=ActionExpert)`` in
``dense`` reading mode, feeds identical random latents / actions / text context, and reports the
max-abs and relative differences of the predicted video and action velocities.

    source scripts/env.sh
    PYTHONPATH="$OPENWAM_VENDOR:$PYTHONPATH" /usr/bin/python3.10 scripts/verify_alpha_equivalence.py \
        --alpha /path/to/OpenWAM-Alpha-Sim-RoboDojo [--device cuda:0] [--steps 2]
"""
from __future__ import annotations

import argparse
import time

import torch

from metiswam4d.config import ExpertSize, ModelConfig
from metiswam4d.build import build_model
from metiswam4d.experts.alpha import load_alpha_action, load_alpha_proprio, load_alpha_video
from metiswam4d.model import ModelInput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--steps", type=int, default=2, help="number of random inputs to compare")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(0)

    t0 = time.time()
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    _, alpha, _ = build_architecture_from_ckpt_dir(args.alpha, weights_required=True, load_text_encoder=False)
    alpha = alpha.to(device=device, dtype=dtype).eval()
    print(f"[openwam] loaded in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    cfg = ModelConfig(experts=("video", "action"), action_read="dense", focus=None,
                      video=ExpertSize(3072, 14336, 30), action=ExpertSize(1024, 4096, 30),
                      gradient_checkpointing=False)
    local = build_model(cfg)
    rv = load_alpha_video(local.video, args.alpha)
    ra = load_alpha_action(local.action, args.alpha)
    rp = load_alpha_proprio(local.proprio_encoder, args.alpha)
    print(f"[local] loaded video={rv['loaded']} action={ra['loaded']} proprio={rp['loaded']} in {time.time() - t0:.0f}s", flush=True)
    local = local.to(device=device, dtype=dtype).eval()

    b = 1
    worst = {}
    for step in range(args.steps):
        latents = torch.randn(b, 48, args.frames, args.height, args.width, device=device, dtype=dtype)
        timestep = torch.tensor([float(torch.randint(1, 1000, (1,)))], device=device, dtype=dtype)
        action_t = torch.tensor([float(torch.randint(1, 1000, (1,)))], device=device, dtype=dtype)
        context = torch.randn(b, 40, 4096, device=device, dtype=dtype)
        context_mask = torch.ones(b, 40, dtype=torch.bool, device=device)
        context_mask[:, 33:] = False
        actions = torch.randn(b, args.action_steps, 80, device=device, dtype=dtype)
        proprio = torch.randn(b, 80, device=device, dtype=dtype)

        with torch.no_grad():
            ref_video, ref_action = alpha(
                actions, action_t, proprio=proprio, latents=latents, timestep=timestep, context=context,
                context_mask=context_mask, first_frame_latents=latents[:, :, :1],
                fuse_vae_embedding_in_latents=True,
            )
            out = local(ModelInput(
                context=context, context_mask=context_mask, video=latents, video_sigma=timestep / 1000.0,
                action=actions, action_sigma=action_t / 1000.0, proprio=proprio[:, None],
                proprio_mask=torch.ones(b, 1, 80, dtype=torch.bool, device=device)))

        for name, ref, mine in (("video", ref_video, out.video_velocity), ("action", ref_action, out.action_velocity)):
            ref32, mine32 = ref.float(), mine.float()
            if name == "video":
                ref32, mine32 = ref32[:, :, 1:], mine32[:, :, 1:]  # the clean frame is not predicted
            diff = (ref32 - mine32).abs()
            rel = diff.max() / ref32.abs().max().clamp(min=1e-6)
            worst[name] = max(worst.get(name, 0.0), float(rel))
            print(f"[step {step}] {name}: max|diff|={diff.max():.4e} mean|diff|={diff.mean():.4e} "
                  f"max|ref|={ref32.abs().max():.3f} rel={float(rel):.3e}", flush=True)
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    status = "OK" if all(v < tol for v in worst.values()) else "MISMATCH"
    print(f"[result] {status} worst relative differences: {worst} (tolerance {tol})")


if __name__ == "__main__":
    main()
