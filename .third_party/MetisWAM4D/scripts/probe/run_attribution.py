#!/usr/bin/env python
"""Gradient-based attribution of the ACTION output to the future-video tokens (OpenWAM-Alpha).

Attention alone can be criticised as "not importance".  Here, at denoising steps ``--steps`` (default 2, 5, 8),
we re-run the model forward with gradients enabled from the states reached by the normal no-grad loop and take

  attn_grad[k]  = sum_{layers, heads, action queries} p_k * d y / d p_k       (attention x gradient, Grad-CAM style)
  latent_sal[k] = || d y / d x_k ||                                           (saliency w.r.t. the noisy future latents)

with y = mean of the predicted action velocity over the active action dims (squared).  Both are reported per
future-video token together with the plain attention, so the three "where does the action read" lenses can be
compared on the same windows (lift on interaction zone / manipulated object / arm; top-5% composition).

    source scripts/probe/env.sh; CUDA_VISIBLE_DEVICES=0 $PY scripts/probe/run_attribution.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Intervention, load_windows  # noqa: E402
from harness_alpha import AlphaProbe  # noqa: E402
from openwam.deploy.denoise_schedule import schedule_sync  # noqa: E402
from run_openloop import window_seed  # noqa: E402

OUT = Path(os.environ.get("PROBE_OUT", "/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes"))


def attribute(probe: AlphaProbe, sample: dict, seed: int, steps: list[int]) -> dict:
    """Returns per requested step: attention [K], attn_grad [K], latent saliency [F-1, h, w]."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    vb, ab = probe.alpha.video_backbone, probe.alpha.action_backbone
    with torch.inference_mode():
        inputs = vb.preprocess_input_for_inference(prompt=sample["prompt"], first_frame_image=list(sample["first_frame_image"]),
                                                   num_frames=9, height=384, width=320, seed=seed, num_inference_steps=probe.steps,
                                                   shift=vb.shift_video, tiled=False, prompt_embed_cache=probe.text_cache)
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}     # leave inference mode
    ref = inputs["first_frame_latents"]
    inputs["latents"][:, :, :1] = ref
    action_noise = torch.randn((1, 32, 80), device=probe.device, dtype=probe.dtype)
    action = action_noise.clone()
    active = sample["proprio_mask"][0].to(probe.device)
    proprio = sample["proprio"].to(probe.device, probe.dtype)[None]
    probe.segments = probe.attn.segments = probe._measure_segments(inputs)
    probe.attn.intervention = Intervention.none()
    probe.attn.capture_kv = False
    schedule = schedule_sync(vb.scheduler, ab.scheduler, probe.steps, shift=ab.shift_action, shift_video=vb.shift_video)
    results = {}
    seg = probe.segments
    a, b = seg.ranges["video_future"]
    for step, ((tv, ta), (tv_next, ta_next)) in enumerate(zip(schedule[:-1], schedule[1:])):
        sv, sa = tv / vb.scheduler.num_train_timesteps, ta / ab.scheduler.num_train_timesteps
        svn, san = tv_next / vb.scheduler.num_train_timesteps, ta_next / ab.scheduler.num_train_timesteps
        call = dict(inputs)
        call["timestep"] = torch.tensor([tv], device=probe.device, dtype=probe.dtype)
        call["_proprio_sample_mask"] = active[None, None]
        want = step in steps
        probe.attn.reset_forward(step)
        probe.attn.capture_attn = want
        probe.attn.keep_grad = want
        if want:
            lat = inputs["latents"].detach().clone().requires_grad_(True)
            call["latents"] = lat
            with torch.enable_grad():
                video_v, action_v = probe.alpha(action.detach(), torch.tensor([ta], device=probe.device, dtype=probe.dtype), proprio=proprio, **call)
                y = (action_v[0][:, active].float() ** 2).mean()
                y.backward()
            layers = probe.attn.attn_layers                                   # list of [B, h, A, K] with grads
            att = torch.stack([p.detach().float() for p in layers])[:, 0].sum(dim=(0, 1, 2))          # [K]
            ag = torch.stack([(p.detach().float() * p.grad.float()) for p in layers])[:, 0].sum(dim=(0, 1, 2))   # [K]
            g = lat.grad[0].float()                                            # [48, F, H, W]
            F_, H_, W_ = g.shape[1:]
            sal = g.norm(dim=0)                                                # [F, H, W]
            sal = sal.reshape(F_, H_ // 2, 2, W_ // 2, 2).sum(dim=(2, 4))[1:]  # [F-1, h, w] per token (patch 2x2)
            results[step] = dict(attention=att.cpu().numpy(), attn_grad=ag.cpu().numpy(), latent_saliency=sal.cpu().numpy())
            video_v, action_v = video_v.detach(), action_v.detach()
            probe.alpha.zero_grad(set_to_none=True)
        else:
            with torch.inference_mode():
                video_v, action_v = probe.alpha(action, torch.tensor([ta], device=probe.device, dtype=probe.dtype), proprio=proprio, **call)
        with torch.no_grad():
            inputs["latents"] = (inputs["latents"] + video_v * (svn - sv)).detach()
            inputs["latents"][:, :, :1] = ref
            action = ab.scheduler.flow_step(action_v, sa, san, action.detach())
            action[..., ~active] = action_noise[..., ~active] * san
        probe.attn.keep_grad = False
        probe.attn.capture_attn = False
    return dict(steps=results, segments=seg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(OUT / "windows.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--steps", default="2,5,8")
    ap.add_argument("--out", default=str(OUT / "attribution" / "alpha"))
    a = ap.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    steps = [int(s) for s in a.steps.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    windows = load_windows(a.windows)[i::n]
    if a.limit:
        windows = windows[: a.limit]
    probe = AlphaProbe()
    probe.alpha.requires_grad_(False)
    t0 = time.time()
    for j, w in enumerate(windows):
        path = out / f"{w['key'].replace('/', '__')}__f{w['start']}.npz"
        if path.exists():
            continue
        seed = window_seed(w["row"], w["start"])
        random.seed(seed)
        sample = probe.dataset.read_window(w["row"], w["start"])
        t = time.time()
        res = attribute(probe, sample, seed, steps)
        seg = res["segments"]
        save = dict(meta=json.dumps(w), segments=json.dumps(dict(ranges=seg.ranges, video_grid=seg.video_grid, track_grid=seg.track_grid)),
                    steps=np.asarray(sorted(res["steps"])))
        for s, r in res["steps"].items():
            for k, v in r.items():
                save[f"{k}_s{s}"] = v.astype(np.float32)
        np.savez_compressed(path, **save)
        print(f"[{j + 1}/{len(windows)}] {w['key']} f{w['start']} {time.time() - t:.0f}s | {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
