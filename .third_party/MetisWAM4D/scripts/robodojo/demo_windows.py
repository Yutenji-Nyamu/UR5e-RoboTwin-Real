"""Spot-check RoboDojo training windows exactly as the trainer sees them (CPU, no VAE).

    /usr/bin/python3.10 scripts/robodojo/demo_windows.py --out <dir> [--n 4] [--split val]

One PNG per window: 9 video frames (3-view L layout, RGB), the 9 Track RGB frames (anchor + 8 uvd transitions,
unknown pixels black), head RGB / robot depth / robot mask conditions, and the 20 base-frame EEF dims
(normalised with Alpha RoboDojo's stats) over the 32 action steps with the proprio row at step 0.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metiswam4d.data.robodojo import RDJEpisodeDataset, RDJWindowConfig
from metiswam4d.data.rt2.eef import SLOTS

NAMES = [f"{arm}_{c}" for arm in ("L", "R") for c in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "grip")]


def strip(frames: torch.Tensor) -> np.ndarray:
    return np.concatenate([f.numpy() for f in frames], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = RDJEpisodeDataset(replace(RDJWindowConfig(), split=args.split, samples_per_episode=1, fixed_windows=True,
                                   seed=args.seed))
    g = torch.Generator().manual_seed(args.seed)
    for j in torch.randperm(len(ds), generator=g)[:args.n].tolist():
        item = ds[j]
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(4, 3, height_ratios=[2.4, 1.8, 1.6, 2.2])
        ax = fig.add_subplot(gs[0, :]); ax.imshow(strip(item["video_frames"])); ax.set_title(f"video frames s+4k (RGB)  {item['key']}"); ax.axis("off")
        ax = fig.add_subplot(gs[1, :]); ax.imshow(strip(item["track_rgb"])); ax.set_title("track RGB: anchor | 8 x (du, dv, dd) mu-law, unknown = black"); ax.axis("off")
        ax = fig.add_subplot(gs[2, 0]); ax.imshow(item["head_rgb"].numpy()); ax.set_title("head RGB condition"); ax.axis("off")
        ax = fig.add_subplot(gs[2, 1]); d = item["head_depth_mm"].numpy() / 1000.0
        ax.imshow(np.where(d > 0, d, np.nan), cmap="viridis"); ax.set_title(f"robot depth m [{d[d > 0].min():.2f}, {d[d > 0].max():.2f}]"); ax.axis("off")
        ax = fig.add_subplot(gs[2, 2]); ax.imshow(item["head_mask"].numpy(), cmap="gray"); ax.set_title(f"robot mask ({item['head_mask'].float().mean():.1%} px)"); ax.axis("off")
        ax = fig.add_subplot(gs[3, :])
        act = torch.cat((item["proprio"], item["action"]), dim=0)[:, SLOTS].numpy()   # [33, 20]
        for k in range(20):
            ax.plot(np.arange(33), act[:, k], label=NAMES[k], lw=1.2, ls="-" if k < 10 else "--")
        ax.axhline(1, c="k", lw=0.5); ax.axhline(-1, c="k", lw=0.5); ax.set_xlim(0, 32)
        ax.set_title("EEF20 in Alpha RoboDojo base frame, min-max normalised (step 0 = proprio)")
        ax.legend(ncol=10, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.suptitle(item["prompt"], fontsize=10)
        name = item["key"].replace("/", "_")
        fig.savefig(out / f"{name}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(out / f"{name}.png", flush=True)


if __name__ == "__main__":
    main()
