#!/usr/bin/env python
"""Statistics for the pixel-frame Track4D representation (du/W, dv/W, dd metres) on the human data and RT2.

Samples IG-10K human episodes (objects: stored RAFT flow + dz; hands: MANO displacement re-projected through
DA3 depth) and Kling hand-track clips (MANO vertices placed and ego-compensated exactly as
``metiswam4d.data.human.hand_render`` does, then projected with the clip intrinsics) and accumulates exact
histograms of |du|/W, |dv|/W, |dd| for

    hand            all hand pixels / vertices
    obj_moving      object pixels with |flow| >= 1.5 px           (real motion)
    obj_quasi       object pixels with 0.5 <= |flow| < 1.5 px     (noise floor of dd: barely moving in the image)
    hand_still      hand pixels / vertices with |du| < 0.5 px     (noise floor of dd on hands)

plus the absolute DA3 depth of valid IG-10K pixels.  Output: per-dataset and pooled quantiles and the
saturation fraction for candidate scales, as JSON + a printed table.  CPU only.

    /usr/bin/python3.10 track_uvd_norm.py --ig10k-per-dir 2 --kling-clips 1500 --workers 12
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

STRIDE = 4
GROUPS = ("hand", "obj_moving", "obj_quasi", "hand_still")
CHANNELS = ("u", "v", "d")
EDGES = {"u": np.logspace(-5, 0, 501), "v": np.logspace(-5, 0, 501), "d": np.logspace(-5, np.log10(3.0), 501)}
DEPTH_EDGES = np.linspace(0, 6.0, 601)
OUT_DIR = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/track_repr_study/uvd_norm")


def empty_hist() -> dict:
    h = {g: {c: np.zeros(len(EDGES[c]) - 1, np.int64) for c in CHANNELS} for g in GROUPS}
    h["depth"] = np.zeros(len(DEPTH_EDGES) - 1, np.int64)
    h["count"] = {g: 0 for g in GROUPS}
    return h


def add(h: dict, group: str, u: np.ndarray, v: np.ndarray, d: np.ndarray) -> None:
    for c, x in zip(CHANNELS, (u, v, d)):
        h[group][c] += np.histogram(np.abs(x), EDGES[c])[0]
    h["count"][group] += int(u.size)


def merge(a: dict, b: dict) -> dict:
    for g in GROUPS:
        for c in CHANNELS:
            a[g][c] += b[g][c]
        a["count"][g] += b["count"][g]
    a["depth"] += b["depth"]
    return a


def quantiles(hist: np.ndarray, edges: np.ndarray, qs=(50, 90, 95, 99, 99.5, 99.9)) -> dict:
    n = hist.sum()
    if n == 0:
        return {}
    cdf = np.cumsum(hist) / n
    return {str(q): float(np.interp(q / 100.0, cdf, edges[1:])) for q in qs}


def saturation(hist: np.ndarray, edges: np.ndarray, scale: float) -> float:
    n = hist.sum()
    return float(hist[edges[1:] > scale].sum() / n) if n else float("nan")


# --------------------------------------------------------------------------- IG-10K
def ig10k_episode(args) -> dict:
    sub, task, ep = args
    import h5py
    from ig10k_anno_reader import IG10KAnno
    h = empty_hist()
    try:
        a = IG10KAnno()
        tr = a.load_track4d(sub, task, ep)
        src, delta, valid, hand, K = tr["src"], tr["delta"], tr["valid"], tr["hand"], tr["K"]
        with h5py.File(a.task_dir(sub, task) / "track4d.h5") as f:
            flow = f[f"ep_{ep:05d}"]["forward_flow_px"][:].astype(np.float32) / 16.0
        P, H, W = valid.shape
        hand_m = hand if hand is not None else np.zeros_like(valid)
        obj = valid & ~hand_m
        fmag = np.linalg.norm(flow, axis=-1)
        dz = delta[..., 2]
        # objects: image displacement is the stored flow, depth displacement is dz (fixed camera)
        mv = obj & (fmag >= 1.5)
        qs = obj & (fmag >= 0.5) & (fmag < 1.5)
        add(h, "obj_moving", flow[..., 0][mv] / W, flow[..., 1][mv] / W, dz[mv])
        add(h, "obj_quasi", flow[..., 0][qs] / W, flow[..., 1][qs] / W, dz[qs])
        depth = a.load_depth(sub, task, ep)
        dep = depth[np.isfinite(depth) & (depth > 0)]
        h["depth"] += np.histogram(dep[:: max(1, dep.size // 2_000_000)], DEPTH_EDGES)[0]
        if hand_m.any():
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            yy, xx = np.mgrid[:H, :W]
            for j in range(P):
                m = hand_m[j]
                if not m.any():
                    continue
                z0 = depth[src[j]][m]
                x0 = (xx[m] - cx) / fx * z0
                y0 = (yy[m] - cy) / fy * z0
                dj = delta[j][m]
                z1 = np.maximum(z0 + dj[:, 2], 1e-3)
                du = (x0 + dj[:, 0]) / z1 * fx + cx - xx[m]
                dv = (y0 + dj[:, 1]) / z1 * fy + cy - yy[m]
                ok = np.isfinite(du) & np.isfinite(dv) & (z0 > 0)
                add(h, "hand", du[ok] / W, dv[ok] / W, dj[ok, 2])
                still = ok & (np.abs(du) < 0.5) & (np.abs(dv) < 0.5)
                add(h, "hand_still", du[still] / W, dv[still] / W, dj[still, 2])
        h["ok"] = f"{task}/ep{ep} pairs={P}"
    except Exception as e:  # noqa: BLE001
        h["err"] = f"{task}/ep{ep}: {type(e).__name__}: {e}"[:200]
    return h


def ig10k_jobs(per_dir: int) -> list[tuple[str, str, int]]:
    import pyarrow.parquet as pq
    rows = pq.read_table("/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/human/manifest.parquet",
                         columns=["subset", "task", "episode", "track4d_h5"]).to_pylist()
    by = {}
    for r in rows:
        if r["track4d_h5"]:
            by.setdefault((r["subset"], r["task"]), []).append(int(r["episode"]))
    jobs = []
    for (sub, task), eps in sorted(by.items()):
        eps = sorted(eps)
        pick = np.unique(np.linspace(0, len(eps) - 1, min(per_dir, len(eps))).round().astype(int))
        jobs += [(sub, task, eps[i]) for i in pick]
    return jobs


# --------------------------------------------------------------------------- Kling
def kling_clip(row: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from metiswam4d.data.human.hand_render import mano_vertices, relative_pose
    from metiswam4d.data.human.kling import load_camera_npz, read_bytes
    from metiswam4d.data.human.window import source_frame_map
    h = empty_hist()
    try:
        W, Hh = int(row["width"]), int(row["height"])
        fit = dict(np.load(io.BytesIO(read_bytes(row["mano_path"], row["mano_offset"], row["mano_size"]))))
        n = int(row["nb_frames"])
        smap = source_frame_map(n, row["src_fps"], int(row["src_nb_frames"]))
        intrinsics = extrinsics = None
        try:
            cam = load_camera_npz(row["camera_npz_archive"], row["camera_npz_member"])
            intrinsics = np.asarray(cam["intrinsics"][0], dtype=np.float64)
            extrinsics = np.asarray(cam["extrinsics"], dtype=np.float64)
        except (KeyError, OSError, tarfile.TarError, ValueError):
            pass
        if intrinsics is not None:
            fx, fy, cx0, cy0 = float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
        else:
            fx = fy = 0.54 * W
            cx0, cy0 = W / 2.0, Hh / 2.0
        K_inv = np.linalg.inv(np.array([[fx, 0, cx0], [0, fy, cy0], [0, 0, 1]], dtype=np.float64))
        rec = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(fit["frame_index"], fit["track_id"]))}
        verts = {}
        idx = np.arange(len(fit["frame_index"]))
        for hand in (False, True):
            sel = idx[fit["is_right"].astype(bool) == hand]
            if len(sel):
                v = mano_vertices(hand, fit["betas"][sel], fit["theta"][sel])
                for i, vi in zip(sel.tolist(), v):
                    verts[i] = vi

        def root(i, depth=None):
            s, cx, cy = (float(x) for x in fit["cam"][i])
            z = fx / max(s, 1e-6) if depth is None else depth
            return (z * (K_inv @ np.array([cx, cy, 1.0]))).astype(np.float32), float(z)

        def project(X):
            z = np.maximum(X[:, 2], 1e-3)
            return np.stack((X[:, 0] / z * fx + cx0, X[:, 1] / z * fy + cy0), -1)

        for k in range(0, n - STRIDE):
            fa, fb = int(smap[k]), int(smap[k + STRIDE])
            if fa == fb:
                continue
            R, t = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            if extrinsics is not None:
                ia, ib = min(fa, len(extrinsics) - 1), min(fb, len(extrinsics) - 1)
                R, _ = relative_pose(extrinsics, ia, ib, True)  # rotation only, as in the loader
            for (f, tid), i in rec.items():
                if f != fa or (fb, tid) not in rec:
                    continue
                kk = rec[(fb, tid)]
                ra, za = root(i)
                rb, _ = root(kk, depth=za)
                xa = ra + verts[i]
                xb = (rb + verts[kk]) @ R.T + t
                pa, pb = project(xa), project(xb)
                du, dv, dd = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1], xb[:, 2] - xa[:, 2]
                add(h, "hand", du / W, dv / W, dd)
                still = (np.abs(du) < 0.5) & (np.abs(dv) < 0.5)
                add(h, "hand_still", du[still] / W, dv[still] / W, dd[still])
        h["ok"] = row["blobstore_key"][-20:]
    except Exception as e:  # noqa: BLE001
        h["err"] = f"{row['blobstore_key'][-20:]}: {type(e).__name__}: {e}"[:200]
    return h


def kling_jobs(n_clips: int) -> list[dict]:
    import pyarrow.parquet as pq
    cols = ["blobstore_key", "width", "height", "nb_frames", "src_fps", "src_nb_frames", "mano_path", "mano_offset",
            "mano_size", "camera_npz_archive", "camera_npz_member"]
    t = pq.read_table("/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M/KlingHumanEgo20M_30fps_hi_handtrack_subset.parquet",
                      columns=cols)
    pick = np.linspace(0, t.num_rows - 1, n_clips).round().astype(int)
    return [{c: t[c][int(i)].as_py() for c in cols} for i in pick]


# --------------------------------------------------------------------------- RT2 (GT depth + K; body = role 1, objects = role 2)
RT2_ROOT = Path("/ytech_milm_intern/danglingwei/datas/RT2_MetisWAM4D")


def rt2_episode(row: dict) -> dict:
    """Groups reuse the slot names: hand = robot body, obj_moving = objects, obj_quasi = objects with < 0.5 px
    image motion, hand_still = body with < 0.5 px image motion (GT floors of dd)."""
    import h5py
    h = empty_hist()
    try:
        d = RT2_ROOT / row["task"] / row["variant"] / f"episode{row['episode']}"
        with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
            n = int(t4d.attrs["frames"])
            for t in range(0, n - STRIDE, 2):
                delta = t4d["delta_xyz_cam"][t].astype(np.float32)
                valid = t4d["valid"][t] > 0
                role = t4d["role"][t]
                depth = src["observation/head_camera/depth"][t].astype(np.float32) / 1000.0
                K = np.asarray(src["observation/head_camera/intrinsic_cv"][t], np.float64)
                H, W = depth.shape
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                yy, xx = np.mgrid[:H, :W]
                fg = valid & (role > 0) & (depth > 0)
                z0 = depth[fg]
                dj = delta[fg]
                x0 = (xx[fg] - cx) / fx * z0
                y0 = (yy[fg] - cy) / fy * z0
                z1 = np.maximum(z0 + dj[:, 2], 1e-3)
                du = (x0 + dj[:, 0]) / z1 * fx + cx - xx[fg]
                dv = (y0 + dj[:, 1]) / z1 * fy + cy - yy[fg]
                dd = dj[:, 2]
                body = role[fg] == 1
                still = (np.abs(du) < 0.5) & (np.abs(dv) < 0.5)
                add(h, "hand", du[body] / W, dv[body] / W, dd[body])
                add(h, "obj_moving", du[~body] / W, dv[~body] / W, dd[~body])
                add(h, "obj_quasi", du[~body & still] / W, dv[~body & still] / W, dd[~body & still])
                add(h, "hand_still", du[body & still] / W, dv[body & still] / W, dd[body & still])
                dep = depth[depth > 0]
                h["depth"] += np.histogram(dep[::8], DEPTH_EDGES)[0]
        h["ok"] = f"{row['task']}/{row['variant']}/ep{row['episode']}"
    except Exception as e:  # noqa: BLE001
        h["err"] = f"{row['task']}/ep{row['episode']}: {type(e).__name__}: {e}"[:200]
    return h


def rt2_jobs(n_episodes: int) -> list[dict]:
    rows = [json.loads(l) for l in open(RT2_ROOT / "index.jsonl") if l.strip()]
    rows = [r for r in rows if r["frames"] >= 33]
    pick = np.linspace(0, len(rows) - 1, min(n_episodes, len(rows))).round().astype(int)
    return [rows[int(i)] for i in np.unique(pick)]


# --------------------------------------------------------------------------- report
def summarize(h: dict, name: str) -> dict:
    out = {"name": name, "count": h["count"], "quantiles": {}, "saturation": {}}
    for g in GROUPS:
        out["quantiles"][g] = {c: quantiles(h[g][c], EDGES[c]) for c in CHANNELS}
    for c, cands in (("u", (1 / 32, 1 / 16, 1 / 8)), ("v", (1 / 32, 1 / 16, 1 / 8)), ("d", (0.05, 0.08, 0.10, 0.15))):
        out["saturation"][c] = {f"{s:.4f}": {g: saturation(h[g][c], EDGES[c], s) for g in ("hand", "obj_moving")} for s in cands}
    if h["depth"].sum():
        out["depth_m"] = quantiles(h["depth"], DEPTH_EDGES, qs=(1, 5, 50, 95, 99))
    return out


LABELS = {"rt2": {"hand": "body(role1)", "obj_moving": "objects", "obj_quasi": "obj<0.5px", "hand_still": "body<0.5px"}}


def print_table(s: dict) -> None:
    labels = LABELS.get(s["name"], {})
    print(f"\n== {s['name']}  counts: " + ", ".join(f"{labels.get(g, g)}={s['count'][g]:,}" for g in GROUPS))
    for g in GROUPS:
        q = s["quantiles"][g]
        if not q["u"]:
            continue
        row = [f"{labels.get(g, g):11s}"]
        for c, unit, k in (("u", "%W", 100.0), ("v", "%W", 100.0), ("d", "mm", 1000.0)):
            row.append(f"{c}[{unit}] p50 {q[c]['50'] * k:7.2f} p90 {q[c]['90'] * k:7.2f} p99 {q[c]['99'] * k:7.2f} p99.5 {q[c]['99.5'] * k:7.2f}")
        print("  " + " | ".join(row))
    for c in CHANNELS:
        print(f"  saturation {c}: " + "  ".join(f"s={k}: hand {v['hand']:.4f} obj {v['obj_moving']:.4f}" if v["obj_moving"] == v["obj_moving"] else f"s={k}: hand {v['hand']:.4f}"
                                            for k, v in s["saturation"][c].items()))
    if "depth_m" in s:
        print("  IG-10K abs depth m: " + ", ".join(f"p{k} {v:.2f}" for k, v in s["depth_m"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ig10k-per-dir", type=int, default=2)
    ap.add_argument("--kling-clips", type=int, default=1500)
    ap.add_argument("--rt2-episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    results = {}
    for name, fn, jobs in (("ig10k", ig10k_episode, ig10k_jobs(args.ig10k_per_dir)),
                           ("kling", kling_clip, kling_jobs(args.kling_clips)),
                           ("rt2", rt2_episode, rt2_jobs(args.rt2_episodes))):
        if not jobs:
            continue
        t0 = time.time()
        total = empty_hist()
        errs = []
        with ctx.Pool(args.workers) as pool:
            for i, h in enumerate(pool.imap_unordered(fn, jobs, chunksize=2)):
                if "err" in h:
                    errs.append(h["err"])
                merge(total, h)
                if (i + 1) % 50 == 0:
                    print(f"  {name}: {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s, {len(errs)} errors)", flush=True)
        s = summarize(total, name)
        s["jobs"] = len(jobs)
        s["errors"] = errs[:20]
        results[name] = (total, s)
        print_table(s)
        if errs:
            print(f"  {len(errs)} errors, first: {errs[:3]}")
        np.savez(args.out / f"{name}_hist.npz", **{f"{g}_{c}": total[g][c] for g in GROUPS for c in CHANNELS}, depth=total["depth"])
    pooled = empty_hist()
    for name in ("ig10k", "kling"):
        if name in results:
            merge(pooled, results[name][0])
    sp = summarize(pooled, "pooled human (ig10k + kling)")
    if pooled["count"]["hand"]:
        print_table(sp)
    (args.out / "uvd_norm_stats.json").write_text(json.dumps(
        {"edges_note": "histograms log-spaced; u = du/W, v = dv/W (frame-width fraction), d metres",
         **{k: v[1] for k, v in results.items()}, "pooled": sp}, indent=1))
    print("written", args.out)


if __name__ == "__main__":
    main()
