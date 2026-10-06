#!/usr/bin/env python
"""Spot-check panels + statistics of the raw human training windows (no GPU).

For every configured component draw N windows and write ``<out>/<component>/<i>_<key>.png`` with the
rows  video frames (9) | Track4D RGB (anchor + 8) | conditions (RGB, depth, mask) and the prompt, plus
``<out>/stats.json`` (per-window timing, hand pairs per transition, displacement saturation, camera
rotation / translation code magnitudes, body / object pixel counts).

    /usr/bin/python3.10 scripts/human_pretrain/demo_windows.py --config configs/stage1_pretrain_human.yaml --n 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from metiswam4d.config import load_stage_config
from metiswam4d.data.camera import camera_motion_magnitude
from metiswam4d.data.human.hand_render import hand_pixels
from metiswam4d.data.human.loader import build_component


def strip(frames: np.ndarray, scale: float = 0.5) -> Image.Image:
    h, w = frames.shape[1:3]
    tw, th = int(w * scale), int(h * scale)
    out = Image.new("RGB", (tw * len(frames), th))
    for i, f in enumerate(frames):
        out.paste(Image.fromarray(np.ascontiguousarray(f)).resize((tw, th), Image.BILINEAR), (i * tw, 0))
    return out


def panel(sample: dict, depth_range=(0.2, 2.5)) -> Image.Image:
    rows = [strip(sample["video_frames"].numpy())]
    if sample.get("track_rgb") is not None:
        rows.append(strip(sample["track_rgb"].numpy()))
        depth = sample["head_depth_m"].numpy()
        d_u8 = np.clip((depth - depth_range[0]) / (depth_range[1] - depth_range[0]), 0, 1) * 255
        cond = np.stack([sample["head_rgb"].numpy(), np.repeat(d_u8[..., None], 3, -1).astype(np.uint8),
                         np.repeat(sample["head_mask"].numpy()[..., None] * 255, 3, -1).astype(np.uint8)])
        rows.append(strip(cond))
    w = max(r.width for r in rows)
    text_h = 44
    out = Image.new("RGB", (w, sum(r.height for r in rows) + text_h), (20, 20, 20))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    draw = ImageDraw.Draw(out)
    prompt = sample.get("prompt", "")
    draw.text((4, y + 2), sample["key"], fill=(255, 255, 0))
    draw.text((4, y + 16), prompt[:150], fill=(230, 230, 230))
    draw.text((4, y + 30), prompt[150:300], fill=(230, 230, 230))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_pretrain_human.yaml")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/stage1_pretrain_human/demo_windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--components", nargs="*", default=None)
    a = ap.parse_args()
    stage = load_stage_config(a.config)
    out = Path(a.out)
    stats: dict = {}
    rng = np.random.default_rng(a.seed)
    for comp in stage.data.human.components:
        if a.components and comp.name not in a.components:
            continue
        ds = build_component(comp)
        folder = out / comp.name
        folder.mkdir(parents=True, exist_ok=True)
        rec: dict = {"n": 0, "seconds": [], "pairs": [], "saturation": [], "cam_rot_deg": [], "cam_trans": [],
                     "body_px": [], "object_px": [], "fg_frames_of_8": [], "prompt_chars": []}
        for i in range(a.n):
            idx = int(rng.integers(len(ds)))
            t0 = time.time()
            s = ds[idx]
            if s.get("hand_verts") is not None:
                px = hand_pixels(s["hand_verts"][None], s["hand_colors"][None], s["hand_is_right"][None],
                                 s["hand_present"][None], *s["video_frames"].shape[1:3])
                s.update({k: v[0] for k, v in px.items()})
            rec["seconds"].append(time.time() - t0)
            rec["prompt_chars"].append(len(s.get("prompt", "")))
            if s.get("track_rgb") is not None:
                roles = s["track_role_px"][1:]
                rec["body_px"].append(int((roles == 1).sum()) / 8)
                rec["object_px"].append(int((roles == 2).sum()) / 8)
                rec["fg_frames_of_8"].append(int(s["track_foreground"][1:].flatten(1).any(1).sum()))
                if s.get("camera_valid") is not None and bool(s["camera_valid"].any()):
                    mag = camera_motion_magnitude(s["camera_delta"][s["camera_valid"]])
                    rec["cam_rot_deg"].append(float(np.degrees(mag[:, 1].mean())))
                    rec["cam_trans"].append(float(mag[:, 0].mean()))
            if s.get("track_pairs") is not None:
                rec["pairs"].append(s["track_pairs"].tolist())
                rec["saturation"].append(float(s["track_saturation"]))
            panel(s).save(folder / f"{i:02d}_{s['key'].replace('/', '_')}.png")
            rec["n"] += 1
            print(f"[{comp.name}] {i:02d} {s['key']} {rec['seconds'][-1]:.2f}s", flush=True)
        summary = {"n": rec["n"], "seconds_median": float(np.median(rec["seconds"])),
                   "seconds_p90": float(np.percentile(rec["seconds"], 90)),
                   "prompt_chars_median": float(np.median(rec["prompt_chars"]))}
        if rec["pairs"]:
            p = np.asarray(rec["pairs"])
            summary.update(pairs_mean_per_transition=float(p.mean()), transitions_with_pair_mean=float((p > 0).sum(1).mean()),
                           saturation_median=float(np.median(rec["saturation"])), saturation_p90=float(np.percentile(rec["saturation"], 90)))
        if rec["body_px"]:
            summary.update(body_px_median=float(np.median(rec["body_px"])), object_px_median=float(np.median(rec["object_px"])),
                           fg_frames_of_8_mean=float(np.mean(rec["fg_frames_of_8"])))
        if rec["cam_rot_deg"]:
            summary.update(cam_rot_deg_median=float(np.median(rec["cam_rot_deg"])), cam_rot_deg_p90=float(np.percentile(rec["cam_rot_deg"], 90)),
                           cam_trans_code_median=float(np.median(rec["cam_trans"])))
        stats[comp.name] = summary
        print(json.dumps({comp.name: summary}, indent=1), flush=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"panels -> {out}")


if __name__ == "__main__":
    main()
