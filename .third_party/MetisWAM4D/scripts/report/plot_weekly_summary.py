"""Report-level training curves (raw + EMA, TensorBoard style): pre-training and post-training.

Each config change is a seam: the EMA restarts there and a labelled vertical line marks it, so jumps
caused by the change stay visible. The y axis is logarithmic and framed on the curve after the
initial transient, so late-stage progress is not flattened by the first few hundred steps.

    /usr/bin/python3.10 scripts/report/plot_weekly_summary.py --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_weekly_curves import ROOT, S2_OFFSET, ema, series, stitch  # noqa: E402

BLUE, ORANGE, GREEN, GRAY = "#3B6FB6", "#E07B39", "#4E9A5B", "#8A8A8A"
O = S2_OFFSET

PRETRAIN_SEAMS = [(14872, "uvd track"), (20543, "track weight x5"), (O, "+ robot data, + action expert"),
                  (O + 2008, "LR up"), (O + 4287, "gripper label"), (O + 6000, "dense read off"),
                  (O + 8434, "co-denoising")]
RT2_SEAMS = [(2328, "uvd track"), (6000, "dense read off"), (30804, "co-denoising")]
RDJ_SEAMS = [(3000, "dense read off"), (13407, "co-denoising")]


def seg_ema(y: np.ndarray, alpha: float, warm: int = 10) -> np.ndarray:
    """EMA seeded with the median of the first points, so a restart does not start from one noisy sample."""
    return ema(np.concatenate([[np.median(y[:warm])], y]), alpha)[1:]


def log_ticks(ax, lo, hi):
    mant = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8])
    cands = np.sort(np.concatenate([mant * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1,
                                                                     int(np.ceil(np.log10(hi))) + 1)]))
    cands = cands[(cands >= lo) & (cands <= hi)]
    if len(cands) > 6:
        idx = np.unique(np.round(np.linspace(0, len(cands) - 1, 6)).astype(int))
        cands = cands[idx]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(cands))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def curve(ax, recs, key, color, title, seams, *, alpha=0.95, skip=0.05):
    x, y = series(recs, key)
    keep = y > 0
    x, y = x[keep], y[keep]
    bounds = [s for s, _ in seams if x[0] < s < x[-1]]
    seg = np.searchsorted(bounds, x, side="right")
    smooth = np.empty_like(y)
    for k in np.unique(seg):
        m = seg == k
        ax.plot(x[m] / 1000, y[m], color=color, alpha=0.2, lw=0.6)
        smooth[m] = seg_ema(y[m], alpha)
        ax.plot(x[m] / 1000, smooth[m], color=color, lw=2.0)
    ax.annotate(f"{smooth[-1]:.3g}", (x[-1] / 1000, smooth[-1]), textcoords="offset points", xytext=(5, 0),
                ha="left", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_yscale("log")
    late = x >= x[0] + skip * (x[-1] - x[0])
    lo, hi = smooth[late].min(), smooth[late].max()
    ax.set_ylim(lo / 1.15, hi * 1.3)
    log_ticks(ax, lo / 1.15, hi * 1.3)
    ax.set_xlim(x[0] / 1000 - 0.02 * (x[-1] - x[0]) / 1000, x[-1] / 1000 + 0.09 * (x[-1] - x[0]) / 1000)

    top = ax.get_ylim()[1]
    visible = [(s, lab) for s, lab in seams if x[0] < s <= x[-1]]
    for i, (s, lab) in enumerate(visible):
        ax.axvline(s / 1000, color=GRAY, ls="--", lw=0.8, alpha=0.8)
        crowded = i + 1 < len(visible) and visible[i + 1][0] - s < 0.06 * (x[-1] - x[0])
        ax.text(s / 1000, top, f" {lab}", rotation=90, va="top", ha="right" if crowded or i == 0 else "left",
                fontsize=7, color="#555")

    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel("training steps (k)", fontsize=9)


def pretrain(path: Path):
    recs = stitch([
        ("stage1_pretrain_human/train_log.jsonl", 0, 14872, 0),
        ("stage1_pretrain_human_uvd/train_log.jsonl", 14872, O + 1, 0),
        ("stage2_pretrain_joint/train_log.jsonl", 0, 8434, O),
        ("stage2_pretrain_joint_v2_coupled/train_log.jsonl", 8434, None, O),
    ])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    curve(axes[0], recs, "loss/track", BLUE, "4D track loss (human + robot)", PRETRAIN_SEAMS)
    curve(axes[1], recs, "loss/video", ORANGE, "Video loss (human + robot)", PRETRAIN_SEAMS)
    curve(axes[2], recs, "loss/action", GREEN, "Robot action loss", PRETRAIN_SEAMS, alpha=0.9)
    fig.suptitle("Pre-training: KlingHumanEgo-2.5M-5000H + IG-10K human / robot", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def posttrain(path: Path):
    rt2 = stitch([
        ("rt2_direct_v2/train_log.jsonl", 0, 2328, 0),
        ("rt2_direct_v2_uvd/train_log.jsonl", 2328, 30804, 0),
        ("rt2_direct_v3_coupled/train_log.jsonl", 30804, None, 0),
    ])
    rdj = stitch([
        ("stage3_robodojo/train_log.jsonl", 0, 13407, 0),
        ("stage3_robodojo_v2_coupled/train_log.jsonl", 13407, None, 0),
    ])
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    for row, (recs, name, seams) in enumerate(((rt2, "RoboTwin2", RT2_SEAMS), (rdj, "RoboDojo", RDJ_SEAMS))):
        curve(axes[row, 0], recs, "loss/action", GREEN, f"{name}: action loss", seams)
        curve(axes[row, 1], recs, "loss/track", BLUE, f"{name}: 4D track loss", seams)
        curve(axes[row, 2], recs, "loss/video", ORANGE, f"{name}: video loss", seams)
    fig.suptitle("Post-training on simulation benchmarks", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "weekly_260929")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pretrain(args.out / "summary_pretrain.png")
    posttrain(args.out / "summary_posttrain.png")
    print("wrote", args.out / "summary_pretrain.png", args.out / "summary_posttrain.png")


if __name__ == "__main__":
    main()
