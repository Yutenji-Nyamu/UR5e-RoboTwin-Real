#!/usr/bin/env python3
"""Qualitative attention figures for the baselines (P2.3 / P2.4).

For a window: Action -> future-world attention (summed over heads, queries and denoising steps unless
``--step``) shown as log2(attention / uniform) on the head-view token grid, per layer group, for both future
latent frames, with the GT interaction zone (cyan) and object mask (white) contours; plus the wrist-token
share and the segment mass vs. area bars.

    python3 scripts/probe/plot_attention.py --model alpha --n 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_openloop import window_gt  # noqa: E402

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
GROUPS = [("L0-9", range(0, 10)), ("L10-19", range(10, 20)), ("L20-29", range(20, 30)), ("all layers", range(30))]


def load(f: Path):
    z = np.load(f, allow_pickle=True)
    meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
    return z, meta, json.loads(str(z["segments"]))


def plot_window(f: Path, out_dir: Path, step: int | None = None) -> Path:
    z, meta, seg = load(f)
    r = seg["ranges"]
    fv, vh, vw = seg["video_grid"]
    A = z["attn_step_layer"]                                   # [S, L, K]
    if step is not None:
        A = A[step:step + 1]
    gt = window_gt(meta["key"], meta["start"])
    rgb = z["pred_rgb"]                                        # [9, 384, 320, 3]
    n_g = len(GROUPS)
    fig = plt.figure(figsize=(3.3 * n_g + 3.5, 8.2))
    gs = fig.add_gridspec(3, n_g + 1, width_ratios=[1] * n_g + [1.1], height_ratios=[1, 1, 0.9])
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for col, (gname, layers) in enumerate(GROUPS):
        a = A[:, list(layers)].sum(axis=(0, 1))
        vf = a[r["video_future"][0]:r["video_future"][1]].reshape(fv - 1, vh, vw)
        vf = vf / vf.sum()
        uniform = 1.0 / vf.size
        ratio = np.log2(np.maximum(vf, 1e-9) / uniform)
        for fi in range(2):
            ax = fig.add_subplot(gs[fi, col])
            frame_img = rgb[4 if fi == 0 else 8][:256]
            ax.imshow(frame_img, alpha=0.9)
            im = ax.imshow(np.kron(ratio[fi, :8], np.ones((32, 32))), cmap="RdBu_r", norm=norm, alpha=0.6, interpolation="nearest")
            inter = np.kron(gt["video"]["inter"][fi] > 0, np.ones((32, 32)))
            obj = np.kron(gt["video"]["obj"][fi] > 0.05, np.ones((32, 32)))
            ax.contour(inter, levels=[0.5], colors="cyan", linewidths=1.5)
            ax.contour(obj, levels=[0.5], colors="white", linewidths=0.8, linestyles="--")
            wrist_share = vf[fi, 8:].sum() / max(vf[fi].sum(), 1e-9)
            ax.set_title(f"{gname}  future f{fi + 1}\nwrist share {wrist_share:.0%}, max/uniform {vf[fi].max() / uniform:.1f}x", fontsize=9)
            ax.axis("off")
    # right column: segment mass vs area, entropy
    ax = fig.add_subplot(gs[0:2, n_g])
    a = A.sum(axis=(0, 1))
    names = [n for n in ("video_clean", "video_future", "track_cond", "track_anchor", "track_future", "action") if r[n][1] > r[n][0]]
    mass = [a[r[n][0]:r[n][1]].sum() / a.sum() for n in names]
    area = [(r[n][1] - r[n][0]) / len(a) for n in names]
    y = np.arange(len(names))
    ax.barh(y - 0.2, area, 0.4, label="area share", color="lightgray")
    ax.barh(y + 0.2, mass, 0.4, label="attention mass", color="tab:red")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    vf_all = a[r["video_future"][0]:r["video_future"][1]]
    p = vf_all / vf_all.sum()
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    ax.set_title(f"Action query mass by segment\nfuture-video norm. entropy {ent:.3f}", fontsize=9)
    # bottom row: colorbar + per-step mass on future video / interaction lift
    ax = fig.add_subplot(gs[2, :])
    S = z["attn_step_layer"].shape[0]
    lifts, masses = [], []
    for s in range(S):
        a_s = z["attn_step_layer"][s].sum(axis=0)
        vf = a_s[r["video_future"][0]:r["video_future"][1]].reshape(fv - 1, vh, vw)
        head = vf[:, :8]
        w = gt["video"]["inter"]
        share = w.sum() / w.size
        lifts.append(((head / head.sum()) * w).sum() / max(share, 1e-6) if share > 0 else np.nan)
        masses.append(vf.sum() / a_s.sum())
    ax.plot(range(S), masses, marker="o", label="mass on future video")
    ax.plot(range(S), lifts, marker="s", label="interaction-zone lift (1 = uniform)")
    ax.axhline(1, color="gray", ls=":")
    ax.set_xlabel("denoising step")
    ax.legend(fontsize=8)
    ax.set_title("per denoising step (all layers)", fontsize=9)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.2])
    fig.colorbar(im, cax=cax, label="log2(attn / uniform)")
    ev = ", ".join(f"{e['type']}@{e['frame']}" for e in meta["events"]) or "no event"
    fig.suptitle(f"{Path(f).parent.name}  |  {meta['key']} f{meta['start']} ({meta['kind']}: {ev})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.91, 0.96))
    out = out_dir / (f.stem + ("" if step is None else f"_step{step}") + ".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alpha")
    ap.add_argument("--root", default=str(OUT / "p31"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--kind", default="event")
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    d = Path(a.root) / a.model
    out_dir = d / "analysis" / "attention_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for f in sorted(d.glob("*.npz")):
        _, meta, _ = load(f)
        if a.kind and meta["kind"] != a.kind:
            continue
        print(plot_window(f, out_dir, a.step))
        done += 1
        if done >= a.n:
            break


if __name__ == "__main__":
    main()
