#!/usr/bin/env python3
"""Qualitative figure: Action -> future-video attention of each baseline on the same windows.

Rows = probe windows (interaction events inside the future), columns = GT frame at the event + one column per
model showing log2(attention / uniform) on the head-view token grid of the latent frame that contains the event,
overlaid on the same GT frame, with the GT interaction zone (cyan) and manipulated object (white) contours.
Attention is summed over heads, action queries, denoising steps and layers.

    python3 scripts/probe/make_attention_figure.py --n 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_openloop import window_gt  # noqa: E402

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
BATCH = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
MODELS = [("effwam", "Efficient-WAM"), ("alpha", "OpenWAM-Alpha"), ("xwam", "X-WAM"), ("flowwam", "FlowWAM")]
KEY_EVENTS = ("contact_start", "motion_onset", "liftoff", "settle")


def head_attention(z, frame_idx: int) -> np.ndarray:
    seg = json.loads(str(z["segments"]))
    r = seg["ranges"]
    fv, vh, vw = seg["video_grid"]
    A = z["attn_step_layer"].sum(axis=(0, 1))
    vf = A[r["video_future"][0]:r["video_future"][1]].reshape(fv - 1, vh, vw)
    head = vf[:, :8]                                   # head view rows (all layouts put the head camera first)
    head = head / head.sum()
    uniform = 1.0 / head.size
    return np.log2(np.maximum(head[frame_idx], 1e-9) / uniform)          # [8, 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tasks", default="", help="comma list to prefer")
    a = ap.parse_args()
    models = [(k, n) for k, n in MODELS if (OUT / "p31" / k).exists() and any((OUT / "p31" / k).glob("*.npz"))]
    common = None
    for k, _ in models:
        names = {f.name for f in (OUT / "p31" / k).glob("*.npz")}
        common = names if common is None else common & names
    cands = []
    for name in sorted(common):
        z = np.load(OUT / "p31" / models[0][0] / name, allow_pickle=True)
        meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
        if meta["kind"] != "event":
            continue
        ev = [e for e in meta["events"] if e["type"] in ("motion_onset", "liftoff")]
        if not ev:
            continue
        cands.append((name, meta, min(e["frame"] for e in ev)))
    prefer = [t for t in a.tasks.split(",") if t]
    if prefer:
        cands = [c for c in cands if c[1]["task"] in prefer] + [c for c in cands if c[1]["task"] not in prefer]
    # spread over tasks
    seen, picks = set(), []
    for c in cands:
        if c[1]["task"] in seen:
            continue
        seen.add(c[1]["task"])
        picks.append(c)
        if len(picks) >= a.n:
            break
    if not picks:
        print("no common event windows yet")
        return
    ncol = 1 + len(models)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.9 * ncol, 2.35 * len(picks)))
    axes = np.atleast_2d(axes)
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for row, (name, meta, ev_frame) in enumerate(picks):
        task, variant, epname = meta["key"].split("/")
        start = meta["start"]
        fidx = 0 if ev_frame <= start + 16 else 1
        show_frame = min(start + (8 if fidx == 0 else 24), 10**9)
        with h5py.File(BATCH / task / variant / epname / "source.hdf5", "r") as src:
            T = src["observation/head_camera/depth"].shape[0]
            show_frame = min(show_frame, T - 1)
            img = cv2.cvtColor(cv2.imdecode(np.frombuffer(src["observation/head_camera/rgb"][show_frame], np.uint8), 1), cv2.COLOR_BGR2RGB)
        gt = window_gt(meta["key"], start)
        inter = np.kron(gt["video"]["inter"][fidx] > 0, np.ones((30, 32)))[:240, :320]
        manip = np.kron(gt["video"]["moving"][fidx] > 0.05, np.ones((30, 32)))[:240, :320]
        ax = axes[row, 0]
        ax.imshow(img)
        ax.contour(inter, levels=[0.5], colors="cyan", linewidths=1.2)
        ax.contour(manip, levels=[0.5], colors="white", linewidths=1.0, linestyles="--")
        ev_types = ",".join(sorted({e["type"] for e in meta["events"] if e["type"] in ("motion_onset", "liftoff")}))
        ax.set_title(f"{task}\nGT frame {show_frame} ({ev_types}@{ev_frame})", fontsize=7.5)
        ax.axis("off")
        for col, (key, mname) in enumerate(models, start=1):
            z = np.load(OUT / "p31" / key / name, allow_pickle=True)
            ratio = head_attention(z, fidx)
            ax = axes[row, col]
            ax.imshow(img, alpha=0.85)
            im = ax.imshow(np.kron(ratio, np.ones((30, 32)))[:240, :320], cmap="RdBu_r", norm=norm, alpha=0.6, interpolation="nearest")
            ax.contour(inter, levels=[0.5], colors="cyan", linewidths=1.2)
            ax.contour(manip, levels=[0.5], colors="white", linewidths=1.0, linestyles="--")
            att = np.exp2(ratio) / np.exp2(ratio).sum()
            i_m = gt["video"]["inter"][fidx] > 0
            o_m = gt["video"]["moving"][fidx] > 0.05
            share_i = att[i_m].sum() if i_m.any() else float("nan")
            share_o = att[o_m].sum() if o_m.any() else float("nan")
            ax.set_title(f"{mname}\ninter {share_i:.0%} (area {i_m.mean():.0%}), object {share_o:.0%} (area {o_m.mean():.0%})", fontsize=7.5)
            ax.axis("off")
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="log2(attention / uniform)")
    fig.suptitle("Action → predicted-future attention (head view, latent frame containing the event; cyan = GT hand–object interaction zone, white = manipulated object)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.91, 0.96))
    out = OUT / "figures" / "fig_attention_maps"
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=170)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
