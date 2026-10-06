#!/usr/bin/env python3
"""Aggregate the P3.1 / P2.3 open-loop results written by ``run_openloop.py``.

Outputs (under ``<out>/p31/<model>/analysis/``):

* ``conditions.csv`` / ``conditions.json``: per intervention, mean / median action deviation from the
  un-intervened run (cm, deg, gripper) and error vs. ground truth, split by event / quiet windows;
* ``attention.json``: Action -> world attention statistics (segment mass vs. area, normalised entropy,
  interaction-zone / object / body lift, correlation with GT coupling transition and motion magnitude,
  attention-sink share), per model, per layer group and denoising step;
* ``per_window.csv``: one row per window with the key numbers.

Token-grid conventions (measured from the harness): video tokens per latent frame form a 12x10 grid over
the 384x320 T-layout, rows 0-7 = head camera (240x320 resized to 256x320), rows 8-11 = wrists; Track tokens
form an 8x10 grid over the 256x320 centre-padded head view.  Latent frame 1 covers raw frames start+4..+16,
frame 2 covers start+20..+32.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
BATCH = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
GT = OUT / "gt_events"
H, W = 240, 320
LAYER_GROUPS = {"early": range(0, 10), "mid": range(10, 20), "late": range(20, 30)}


# ----------------------------------------------------------------------------- GT on token grids


def pool_area(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Mean of ``mask[..., H, W]`` over pixel boxes given by row/col edge arrays -> ``[..., R, C]``."""
    out = np.zeros(mask.shape[:-2] + (len(rows) - 1, len(cols) - 1), dtype=np.float32)
    for i in range(len(rows) - 1):
        r0, r1 = max(rows[i], 0), min(rows[i + 1], mask.shape[-2])
        for j in range(len(cols) - 1):
            c0, c1 = cols[j], cols[j + 1]
            if r1 > r0:
                out[..., i, j] = mask[..., r0:r1, c0:c1].mean(axis=(-1, -2))
    return out


VIDEO_HEAD_ROWS = np.round(np.arange(9) * 30).astype(int)           # 8 head token rows over 240 px
VIDEO_COLS = np.arange(11) * 32
TRACK_ROWS = np.arange(9) * 32 - 8                                    # centre pad 8 px
TRACK_COLS = np.arange(11) * 32


def window_gt(key: str, start: int) -> dict:
    """Per latent future frame (2) token maps for head-view video tokens (8x10) and track tokens (8x10)."""
    task, variant, name = key.split("/")
    ep = BATCH / task / variant / name
    z = np.load(GT / task / variant / f"{name}.npz", allow_pickle=False)
    T = int(z["frames"])
    with h5py.File(ep / "masks.h5", "r") as m:
        bits = m["head_camera/mask_bits"]
        frames = [min(start + 4 * k, T - 1) for k in range(1, 9)]
        mb = np.unpackbits(bits[frames], axis=-1, bitorder="little")[..., :W].astype(bool)   # [8, 2, H, W]
    body, obj = mb[:, 0], mb[:, 1]
    inter = np.unpackbits(z["interaction_pix"][frames], axis=-1, bitorder="little")[..., :W].astype(bool)
    manip_pix = np.unpackbits(z["moving_pix"][frames], axis=-1, bitorder="little")[..., :W].astype(bool)

    def grids(rows, cols):
        g = {}
        for nm, px in (("body", body), ("obj", obj), ("inter", inter), ("moving", manip_pix)):
            p = pool_area(px, rows, cols)                                # [8, 8, 10]
            g[nm] = np.stack([p[:4].mean(0), p[4:].mean(0)])            # [2 latent frames, 8, 10]
        return g

    out = dict(video=grids(VIDEO_HEAD_ROWS, VIDEO_COLS), track=grids(TRACK_ROWS, TRACK_COLS))
    # coupling transition |dc| and motion magnitude from the GT Track4D at this window's phase
    phase = start % 4
    src = z[f"p{phase}_src"]
    rows = np.searchsorted(src, [start + 4 * k for k in range(8)])
    rows = np.clip(rows, 0, len(src) - 1)
    dc = np.abs(z[f"p{phase}_dc"][rows].astype(np.float32))            # [8, 15, 20] (16 px grid)
    mag = np.linalg.norm(z[f"p{phase}_disp"][rows].astype(np.float32), axis=-1)
    ro = z["role_obj"][[min(start + 4 * k, T - 1) for k in range(8)]].astype(np.float32)
    dc_obj, mag_obj = dc * ro, mag * ro

    def regrid16(a):   # [8, 15, 20] (16 px) -> [2, 8, 10] (approx 32 px) by 2x2 mean with last-row padding
        a = np.concatenate([a, a[:, -1:]], axis=1)                    # 16 rows
        a = a.reshape(8, 8, 2, 10, 2).mean(axis=(2, 4))
        return np.stack([a[:4].mean(0), a[4:].mean(0)])

    out["dc"] = regrid16(dc_obj)
    out["mag"] = regrid16(mag_obj)
    out["dc_all"] = regrid16(dc)
    out["mag_all"] = regrid16(mag)
    return out


# ----------------------------------------------------------------------------- attention stats


def entropy_norm(p: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(p)))


def lift(att: np.ndarray, weight: np.ndarray) -> float:
    """Attention mass on tokens weighted by ``weight`` (0..1 area fraction) divided by the weighted area share."""
    a = att / max(att.sum(), 1e-12)
    share = weight.sum() / weight.size
    return float((a * weight).sum() / max(share, 1e-6)) if share > 0 else float("nan")


def attention_stats(A: np.ndarray, seg: dict, gt: dict) -> dict:
    """``A`` = attention of the action queries summed over heads and queries: ``[K]`` (one layer, one step)."""
    r = seg["ranges"]
    f, vh, vw = seg["video_grid"]
    stats = {}
    total = A.sum()
    for name in ("video_clean", "video_future", "track_cond", "track_anchor", "track_future", "action"):
        a, b = r[name]
        if b > a:
            stats[f"mass_{name}"] = float(A[a:b].sum() / total)
            stats[f"area_{name}"] = (b - a) / len(A)
    # video future: [2 frames, 12, 10]; head rows 0..7
    a, b = r["video_future"]
    vf = A[a:b].reshape(f - 1, vh, vw)
    head = vf[:, :8]
    stats["video_future_entropy"] = entropy_norm(vf.reshape(-1))
    stats["video_future_head_share"] = float(head.sum() / max(vf.sum(), 1e-12))
    stats["video_future_frame2_share"] = float(vf[1].sum() / max(vf.sum(), 1e-12))
    stats["video_future_sink_top1"] = float(vf.max() / max(vf.sum(), 1e-12))
    g = gt["video"]
    for nm in ("inter", "obj", "body", "moving"):
        stats[f"video_lift_{nm}"] = lift(head, g[nm])
    for nm in ("dc", "mag"):
        x, y = gt[nm].reshape(-1), head.reshape(-1)
        stats[f"video_corr_{nm}"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    a, b = r["track_future"]
    if b > a:
        tf_, th, tw = seg["track_grid"]
        tf = A[a:b].reshape(tf_ - 1, th, tw)
        stats["track_future_entropy"] = entropy_norm(tf.reshape(-1))
        stats["track_future_frame2_share"] = float(tf[1].sum() / max(tf.sum(), 1e-12))
        stats["track_future_sink_top1"] = float(tf.max() / max(tf.sum(), 1e-12))
        g = gt["track"]
        for nm in ("inter", "obj", "body", "moving"):
            stats[f"track_lift_{nm}"] = lift(tf, g[nm])
        for nm in ("dc", "mag"):
            x, y = gt[nm].reshape(-1), tf.reshape(-1)
            stats[f"track_corr_{nm}"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return stats


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="janus")
    ap.add_argument("--root", default=str(OUT / "p31"))
    a = ap.parse_args()
    d = Path(a.root) / a.model
    files = sorted(d.glob("*.npz"))
    out_dir = d / "analysis"
    out_dir.mkdir(exist_ok=True)
    cond_rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    attn_acc: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    per_window = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = z["meta"]
        meta = meta.item() if meta.dtype == object else json.loads(str(meta))
        errors = json.loads(str(z["errors"]))
        seg = json.loads(str(z["segments"]))
        kind = meta["kind"]
        row = dict(key=meta["key"], start=meta["start"], task=meta["task"], kind=kind)
        for cond, e in errors.items():
            for ref in ("vs_full", "vs_gt"):
                for m in ("pos_cm", "rot_deg", "grip", "norm_l2", "joint_deg"):
                    val = e[ref].get(m, float("nan"))
                    cond_rows[cond][f"{ref}_{m}_{kind}"].append(val)
                    cond_rows[cond][f"{ref}_{m}_all"].append(val)
            row[f"{cond}_pos"] = e["vs_full"]["pos_cm"]
            row[f"{cond}_joint"] = e["vs_full"].get("joint_deg", float("nan"))
        # attention
        try:
            gt = window_gt(meta["key"], meta["start"])
        except Exception as exc:  # noqa: BLE001
            print("gt failed", meta["key"], exc)
            gt = None
        if gt is not None:
            A_sl = z["attn_step_layer"]                                 # [S, L, K]
            S, L = A_sl.shape[0], A_sl.shape[1]
            # layer groups scale with depth (Alpha/X-WAM/FlowWAM 30 layers, Efficient-WAM 12)
            layer_groups = {"early": range(0, L // 3), "mid": range(L // 3, 2 * L // 3), "late": range(2 * L // 3, L)}
            for step_name, s in (("first", 0), ("mid", S // 2), ("last", S - 1), ("mean", None)):
                for grp, layers in layer_groups.items():
                    A = A_sl[:, list(layers)].sum(axis=1) if s is None else A_sl[s, list(layers)].sum(axis=0)
                    if s is None:
                        A = A.sum(axis=0)
                    st = attention_stats(A, seg, gt)
                    for k, v in st.items():
                        attn_acc[(step_name, grp)][k].append(v)
                        attn_acc[(step_name, grp)][f"{k}__{kind}"].append(v)
            A_all = A_sl.sum(axis=(0, 1))
            st = attention_stats(A_all, seg, gt)
            row.update({f"attn_{k}": v for k, v in st.items() if k.startswith(("mass_", "video_lift", "track_lift", "video_future_entropy", "track_future_entropy"))})
        per_window.append(row)

    def summarize(values):
        v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
        return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                    p90=float(np.percentile(v, 90)) if len(v) else None)

    conditions = {c: {k: summarize(v) for k, v in m.items()} for c, m in cond_rows.items()}
    (out_dir / "conditions.json").write_text(json.dumps(conditions, indent=1))
    # primary deviation metric: EEF cm for EEF models, joint degrees for joint-space models
    M = "pos_cm" if any(np.isfinite(v) for v in cond_rows.get("full", {}).get("vs_gt_pos_cm_all", [])) else "joint_deg"
    with open(out_dir / "conditions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "n", "vs_full_pos_cm_mean", "vs_full_pos_cm_median", "vs_full_rot_deg_mean", "vs_full_grip_mean",
                    "vs_gt_pos_cm_mean", "event_vs_full_pos_mean", "quiet_vs_full_pos_mean"])
        for c, m in sorted(conditions.items()):
            w.writerow([c, m[f"vs_full_{M}_all"]["n"], m[f"vs_full_{M}_all"]["mean"], m[f"vs_full_{M}_all"]["median"],
                        m["vs_full_rot_deg_all"]["mean"], m["vs_full_grip_all"]["mean"], m[f"vs_gt_{M}_all"]["mean"],
                        m.get(f"vs_full_{M}_event", {}).get("mean"), m.get(f"vs_full_{M}_quiet", {}).get("mean")])
    (out_dir / "primary_metric.txt").write_text(M)
    attention = {f"{s}/{g}": {k: summarize(v) for k, v in m.items()} for (s, g), m in attn_acc.items()}
    (out_dir / "attention.json").write_text(json.dumps(attention, indent=1))
    if per_window:
        keys = sorted({k for r in per_window for k in r})
        with open(out_dir / "per_window.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(per_window)

    print(f"{len(files)} windows, model={a.model}, primary metric = {M}")
    print(f"{'condition':24s} {'n':>4s} {'dev mean/med':>20s} {'grip':>6s} {'normL2':>7s} | {'vsGT':>8s} | event / quiet")
    fmt = lambda v: f"{v:.2f}" if v is not None else "  nan"
    for c, m in sorted(conditions.items(), key=lambda kv: kv[1][f"vs_full_{M}_all"]["mean"] or 0):
        x = m[f"vs_full_{M}_all"]
        print(f"{c:24s} {x['n']:4d} {fmt(x['mean']):>9s}/{fmt(x['median']):>9s} {fmt(m['vs_full_grip_all']['mean']):>6s} {fmt(m['vs_full_norm_l2_all']['mean']):>7s} | "
              f"{fmt(m[f'vs_gt_{M}_all']['mean']):>8s} | {fmt(m.get(f'vs_full_{M}_event', {}).get('mean'))} / {fmt(m.get(f'vs_full_{M}_quiet', {}).get('mean'))}")
    print("\nattention (mean over steps, all layers):")
    key = "mean/mid"
    for grp in ("early", "mid", "late"):
        m = attention.get(f"mean/{grp}", {})
        if not m:
            continue
        line = [f"[{grp}]"]
        for k in ("mass_video_clean", "mass_video_future", "mass_track_future", "mass_action", "video_future_entropy",
                  "video_lift_inter", "video_lift_obj", "video_lift_body", "video_corr_dc", "video_corr_mag",
                  "track_future_entropy", "track_lift_inter", "track_lift_obj", "track_corr_dc", "track_corr_mag", "video_future_sink_top1"):
            if k in m and m[k]["mean"] is not None:
                line.append(f"{k}={m[k]['mean']:.3f}")
        print(" ".join(line))


if __name__ == "__main__":
    main()
