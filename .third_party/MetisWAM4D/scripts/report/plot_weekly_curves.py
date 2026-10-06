"""Stitch resumed runs into one lineage per track and plot curves with the config changes marked.

Each `train_log.jsonl` is appended across resumes; a step going backwards is a resume, so later
records replace everything from that step on (abandoned branches drop out). Segments of different
runs are then concatenated at the checkpoint the next run was initialized from.

    /usr/bin/python3.10 scripts/report/plot_weekly_curves.py --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921")
TZ = dt.timezone(dt.timedelta(hours=8))
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#DD8452", "#937860"]


def read_lineage(path: Path) -> dict[int, dict]:
    kept: list[dict] = []
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = rec["step"]
        if kept and step < kept[-1]["step"]:
            kept = [r for r in kept if r["step"] < step]
        kept.append(rec)
    merged: dict[int, dict] = {}
    for rec in kept:
        merged.setdefault(rec["step"], {}).update(rec)
    return merged


def stitch(segments) -> dict[int, dict]:
    """segments: [(log path, first step, last step exclusive or None, global offset)]."""
    out: dict[int, dict] = {}
    for path, lo, hi, offset in segments:
        for step, rec in read_lineage(ROOT / path).items():
            if step >= lo and (hi is None or step < hi):
                out[step + offset] = rec
    return dict(sorted(out.items()))


def series(recs: dict[int, dict], key: str):
    xs = [s for s, r in recs.items() if key in r and r[key] is not None and np.isfinite(r[key])]
    return np.array(xs), np.array([recs[s][key] for s in xs], dtype=float)


def date_at(recs: dict[int, dict], step: int) -> str:
    """Wall-clock time of the first record logged under the config that starts at `step`."""
    first = min(s for s in recs if "time" in recs[s] and s > step)
    return dt.datetime.fromtimestamp(recs[first]["time"], TZ).strftime("%m-%d %H:%M")


def phase_of(phases, x):
    idx = 0
    for i, (start, _) in enumerate(phases):
        if x >= start:
            idx = i
    return idx


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc
    return out


def plot_metric(ax, recs, phases, key, *, smooth=None, label=None, style="-", marker=None, lw=1.6):
    """Colour by phase; EMA is restarted at every phase boundary so smoothing never crosses a change."""
    xs, ys = series(recs, key)
    if len(xs) == 0:
        return
    pid = np.array([phase_of(phases, x) for x in xs])
    for p in np.unique(pid):
        m = pid == p
        x, y = xs[m], ys[m]
        c = COLORS[p % len(COLORS)]
        if smooth:
            ax.plot(x, y, color=c, alpha=0.15, lw=0.8)
            y = ema(y, smooth)
        ax.plot(x, y, style, color=c, lw=lw, marker=marker, ms=3,
                label=label if p == pid[0] else None)


def mark_events(ax, events, *, text=False):
    ymin, ymax = ax.get_ylim()
    for x, lab, kind in events:
        ls = ":" if kind == "minor" else "--"
        ax.axvline(x, color="k" if kind != "codec" else "#C44E52", ls=ls, lw=0.8, alpha=0.6)
        if text:
            ax.text(x, ymax, " " + lab, rotation=90, va="top", ha="left", fontsize=6.5, alpha=0.8)


def phase_legend(fig, phases, recs, ncol=2):
    handles = [plt.Line2D([], [], color=COLORS[i % len(COLORS)], lw=3) for i in range(len(phases))]
    labels = [f"[{date_at(recs, start)}] {lab}" for start, lab in phases]
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=7.5, frameon=False)


def finish(fig, axes, events, path, bottom, no_events=()):
    for ax in axes.flat:
        if not ax.has_data():
            continue
        if ax not in no_events:
            mark_events(ax, events, text=(ax is axes.flat[0]))
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------------------------
# Pre-training: S1 human (Video + Track) -> S2 joint (Kling + IG-10K human + robot, three experts)

S2_OFFSET = 21609


def pretrain(out: Path):
    recs = stitch([
        ("stage1_pretrain_human/train_log.jsonl", 0, 14872, 0),
        ("stage1_pretrain_human_uvd/train_log.jsonl", 14872, S2_OFFSET + 1, 0),
        ("stage2_pretrain_joint/train_log.jsonl", 0, 8434, S2_OFFSET),
        ("stage2_pretrain_joint_v2_coupled/train_log.jsonl", 8434, None, S2_OFFSET),
    ])
    o = S2_OFFSET
    phases = [
        (0, "S1 human: Kling hand + IG-10K human | Track4D xyz | w_track 0.2 | LR V3e-7 T3e-6"),
        (14872, "S1: Track4D -> pixel uvd codec"),
        (20543, "S1: w_track 0.2 -> 1.0"),
        (o, "S2 joint: + IG-10K robot, + Action expert (Alpha Foundation) | LR V3e-7 T3e-6 A3e-6"),
        (o + 2008, "S2: LR V1e-6 T1e-5 A1e-5"),
        (o + 4287, "S2: gripper label +-2-step ramp"),
        (o + 8434, "S2 v2: Track-Action co-clock + dense-read floor + read-action probe | LR V1e-5 A2e-5"),
    ]
    events = [(14872, "xyz->uvd", "codec"), (20543, "w_track 1.0", "major"), (o, "S2 start", "major"),
              (o + 2008, "LR up", "major"), (o + 4287, "gripper ramp", "major"),
              (o + 8434, "co-clock", "major")]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axes[0, 0]
    plot_metric(ax, recs, phases, "val/track", marker="o")
    ax.set_title("val/track  (IG-10K human held-out, 1024 windows)\nred line: codec change, not comparable across", fontsize=9)
    ax = axes[0, 1]
    plot_metric(ax, recs, phases, "val/video", marker="o")
    ax.set_title("val/video  (IG-10K human held-out)", fontsize=9)
    ax = axes[0, 2]
    plot_metric(ax, recs, phases, "val/track_object", marker="o", label="object")
    plot_metric(ax, recs, phases, "val/track_body", marker="s", style="--", label="body (hand)")
    ax.set_title("val/track by region  (human held-out)", fontsize=9)
    ax = axes[1, 0]
    plot_metric(ax, recs, phases, "val_robot/action", marker="o", label="total")
    plot_metric(ax, recs, phases, "val_robot/action_xyz", style="--", lw=1.0, label="xyz")
    plot_metric(ax, recs, phases, "val_robot/action_rot6d", style=":", lw=1.0, label="rot6d")
    ax.set_title("val_robot/action  (IG-10K robot held-out, 420 eps)\ngripper label changed at 'gripper ramp'; sigma dist. changed at 'co-clock'", fontsize=9)
    ax = axes[1, 1]
    plot_metric(ax, recs, phases, "val_robot/track", marker="o", label="track")
    plot_metric(ax, recs, phases, "val_robot/video", marker="s", style="--", label="video")
    ax.set_title("val_robot/track & video  (IG-10K robot held-out)", fontsize=9)
    ax = axes[1, 2]
    plot_metric(ax, recs, phases, "focus/read_ratio", smooth=0.9, label="read_ratio (Action->world)")
    plot_metric(ax, recs, phases, "loss/read_action", smooth=0.9, style="--", label="read_action probe loss")
    ax.set_yscale("log")
    ax.set_title("Action world-interface diagnostics (train, EMA)", fontsize=9)
    for a in axes.flat:
        a.set_xlim(0, max(recs) + 300)
        a.set_xlabel(f"global step  (S1 128 win/step; S2 256 win/step, starts at {o})", fontsize=7.5)
    for a in axes[1, :2]:
        a.set_xlim(o - 300, max(recs) + 300)
        a.set_xlabel("global step (S2 only)", fontsize=7.5)
    axes[1, 2].set_xlim(o - 300, max(recs) + 300)
    axes[1, 2].set_xlabel("global step (S2 only)", fontsize=7.5)
    fig.suptitle("Pre-training lineage: S1 human video+Track4D  ->  S2 joint human+robot (three experts)", fontsize=12)
    phase_legend(fig, phases, recs)
    finish(fig, axes, events, out / "fig_pretrain_curves.png", bottom=0.1)
    return recs, phases


# ----------------------------------------------------------------------------------------------
# RoboTwin2 direct training

LEGACY = ROOT / "probes/legacy_action_loss"


def legacy_panel(ax, entries, ref_lo, ref_hi, ref_label):
    names, vals, lows, mids, highs = [], [], [], [], []
    for name, fn in entries:
        d = json.loads((LEGACY / fn).read_text())
        names.append(name)
        vals.append(d["action"])
        lows.append(d["bins"]["low"])
        mids.append(d["bins"]["mid"])
        highs.append(d["bins"]["high"])
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, vals, w, label="all (256 win)", color="#4C72B0")
    ax.bar(x - 0.5 * w, lows, w, label="low sigma", color="#DD8452")
    ax.bar(x + 0.5 * w, mids, w, label="mid sigma", color="#55A868")
    ax.bar(x + 1.5 * w, highs, w, label="high sigma", color="#8172B2")
    for xi, v in zip(x, vals):
        ax.text(xi - 1.5 * w, v, f"{v * 1e3:.2f}e-3", ha="center", va="bottom", fontsize=6.5)
    ax.axhspan(ref_lo, ref_hi, color="gray", alpha=0.3, label=ref_label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_yscale("log")
    ax.set_title("Action loss under the old JanusAct4D protocol\n(sigma_A = sigma_T = shift5(U), no dropout; comparable across runs)", fontsize=9)


def rt2(out: Path):
    recs = stitch([
        ("rt2_direct_v2/train_log.jsonl", 0, 2328, 0),
        ("rt2_direct_v2_uvd/train_log.jsonl", 2328, 30804, 0),
        ("rt2_direct_v3_coupled/train_log.jsonl", 30804, None, 0),
    ])
    phases = [
        (0, "v2: Alpha RT2-Full + Fusion-v3 init, K=32 compact read | async r_A0.25<r_T0.75<r_V1 | Track4D xyz"),
        (2328, "v2_uvd: Track4D -> pixel uvd codec (same recipe; 16 GPU, from 25003 single node 8 GPU)"),
        (30804, "v3_coupled: Track-Action co-clock + dense-read floor + read-action probe | LR V5e-6 T1e-5 A2e-5"),
    ]
    events = [(2328, "xyz->uvd", "codec"), (25003, "16->8 GPU", "minor"), (30804, "co-clock", "major")]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    plot_metric(ax, recs, phases, "loss/action", smooth=0.95)
    ax.set_yscale("log")
    ax.set_title("train loss/action (EMA)\nsigma_A distribution changes at 'co-clock' -> not comparable across", fontsize=9)
    ax = axes[0, 1]
    plot_metric(ax, recs, phases, "loss/action_shortcut_kept", smooth=0.95, label="text+proprio kept")
    plot_metric(ax, recs, phases, "loss/action_shortcut_dropped", smooth=0.95, style="--", label="text/proprio dropped")
    ax.set_yscale("log")
    ax.set_title("action loss: shortcut kept vs dropped (EMA)", fontsize=9)
    ax = axes[0, 2]
    plot_metric(ax, recs, phases, "loss/track", smooth=0.95, label="track")
    plot_metric(ax, recs, phases, "loss/track_body", smooth=0.95, style="--", lw=1.0, label="body")
    plot_metric(ax, recs, phases, "loss/track_object", smooth=0.95, style=":", lw=1.0, label="object")
    ax.set_title("train loss/track (EMA)", fontsize=9)
    ax = axes[1, 0]
    plot_metric(ax, recs, phases, "loss/video", smooth=0.95)
    ax.set_title("train loss/video (EMA)", fontsize=9)
    ax = axes[1, 1]
    plot_metric(ax, recs, phases, "focus/read_ratio", smooth=0.95, label="read_ratio (Action->world)")
    plot_metric(ax, recs, phases, "loss/read_action", smooth=0.9, style="--", label="read_action probe loss")
    ax.set_yscale("log")
    ax.set_title("Action world-interface diagnostics (EMA)", fontsize=9)
    for a in axes.flat[:5]:
        a.set_xlabel("step (128 windows/step)", fontsize=7.5)
    legacy_panel(axes[1, 2], [("v2_uvd\nstep 30804\n(before)", "rt2_v2_uvd_step30804.json"),
                              ("v3_coupled\nstep 32271\n(+1467 steps)", "rt2_v3_coupled_step32271.json")],
                 0.00045, 0.00052, "old JanusAct4D rt2_v1 19k-30k")
    fig.suptitle("RoboTwin2 (27.5k episodes) direct training lineage", fontsize=12)
    phase_legend(fig, phases, recs, ncol=1)
    finish(fig, axes, events, out / "fig_rt2_curves.png", bottom=0.09, no_events=(axes[1, 2],))
    return recs, phases


# ----------------------------------------------------------------------------------------------
# RoboDojo post-training

def robodojo(out: Path):
    recs = stitch([
        ("stage3_robodojo/train_log.jsonl", 0, 13407, 0),
        ("stage3_robodojo_v2_coupled/train_log.jsonl", 13407, None, 0),
    ])
    phases = [
        (0, "stage3: Video/Action <- Alpha RoboDojo, Track+interface <- RT2 uvd 25003 | async | LR V3e-7 T3e-6 A3e-6"),
        (13407, "v2_coupled: Track-Action co-clock + dense-read floor + read-action probe | LR V5e-6 T1e-5 A2e-5"),
    ]
    events = [(13407, "co-clock", "major")]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    plot_metric(ax, recs, phases, "val/action", marker="o", label="total")
    plot_metric(ax, recs, phases, "val/action_xyz", style="--", lw=1.0, label="xyz")
    plot_metric(ax, recs, phases, "val/action_rot6d", style=":", lw=1.0, label="rot6d")
    plot_metric(ax, recs, phases, "val/action_gripper", style="-.", lw=1.0, label="gripper")
    ax.set_yscale("log")
    ax.set_title("val/action  (102 held-out eps, 256 windows)\nsigma_A distribution changes at 'co-clock'", fontsize=9)
    ax = axes[0, 1]
    plot_metric(ax, recs, phases, "loss/action_shortcut_kept", smooth=0.95, label="text+proprio kept")
    plot_metric(ax, recs, phases, "loss/action_shortcut_dropped", smooth=0.95, style="--", label="text/proprio dropped")
    ax.set_yscale("log")
    ax.set_title("train action loss: shortcut kept vs dropped (EMA)", fontsize=9)
    ax = axes[0, 2]
    plot_metric(ax, recs, phases, "val/track_body", marker="o")
    ax.set_title("val/track_body  (robot pixels, FK uvd; objects unsupervised)", fontsize=9)
    ax = axes[1, 0]
    plot_metric(ax, recs, phases, "val/video", marker="o")
    ax.set_title("val/video", fontsize=9)
    ax = axes[1, 1]
    plot_metric(ax, recs, phases, "focus/read_ratio", smooth=0.95, label="read_ratio (Action->world)")
    plot_metric(ax, recs, phases, "loss/read_action", smooth=0.9, style="--", label="read_action probe loss")
    ax.set_yscale("log")
    ax.set_title("Action world-interface diagnostics (EMA)", fontsize=9)
    for a in axes.flat[:5]:
        a.set_xlabel("step (128 windows/step)", fontsize=7.5)
    legacy_panel(axes[1, 2], [("v2_coupled\nstep 14876\n(+1469 steps)", "rdj_v2_coupled_step14876.json")],
                 0.0005, 0.0006, "old JanusAct4D rdj_v1 30k-40k")
    fig.suptitle("RoboDojo post-training lineage (3,400 eps / 34 tasks)", fontsize=12)
    phase_legend(fig, phases, recs, ncol=1)
    finish(fig, axes, events, out / "fig_robodojo_curves.png", bottom=0.08, no_events=(axes[1, 2],))
    return recs, phases


# ----------------------------------------------------------------------------------------------
# Wall-clock overview: which config ran when, per track

def timeline(out: Path, tracks):
    fig, ax = plt.subplots(figsize=(15, 4.2))
    for row, (name, recs, phases, shorts) in enumerate(tracks):
        steps = [s for s in recs if "time" in recs[s]]
        bounds = [p[0] for p in phases] + [max(steps) + 1]
        for i, (start, _) in enumerate(phases):
            seg = [s for s in steps if bounds[i] <= s < bounds[i + 1]]
            if not seg:
                continue
            t0 = dt.datetime.fromtimestamp(recs[seg[0]]["time"], TZ)
            t1 = dt.datetime.fromtimestamp(recs[seg[-1]]["time"], TZ)
            c = COLORS[i % len(COLORS)]
            ax.barh(row, (t1 - t0).total_seconds() / 86400, left=matplotlib.dates.date2num(t0),
                    height=0.55, color=c, alpha=0.85, edgecolor="white")
            above = i % 2 == 1
            ax.text(matplotlib.dates.date2num(t0) + 0.01, row + (-0.32 if above else 0.32), shorts[i],
                    fontsize=6.5, va="bottom" if above else "top", color=c)
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels([t[0] for t in tracks], fontsize=9)
    ax.invert_yaxis()
    ax.xaxis_date(tz=TZ)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d", tz=TZ))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(tz=TZ))
    ax.set_xlim(matplotlib.dates.date2num(dt.datetime(2026, 9, 23, tzinfo=TZ)),
                matplotlib.dates.date2num(dt.datetime(2026, 9, 30, 12, tzinfo=TZ)))
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Training runs this week (each colour = one config segment on the surviving lineage)", fontsize=10)
    fig.tight_layout()
    path = out / "fig_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "weekly_260929")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = pretrain(args.out)
    r = rt2(args.out)
    d = robodojo(args.out)
    timeline(args.out, [
        ("Pre-train", *p, ["S1 human, xyz", "S1 uvd", "w_track 1.0", "S2 joint +robot", "LR up",
                           "gripper ramp", "co-clock"]),
        ("RoboTwin2", *r, ["v2 xyz", "v2_uvd (16 GPU -> 8 GPU at 25k)", "v3 co-clock"]),
        ("RoboDojo", *d, ["stage3 async", "v2 co-clock"]),
    ])


if __name__ == "__main__":
    main()
