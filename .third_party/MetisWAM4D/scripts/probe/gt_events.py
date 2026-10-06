#!/usr/bin/env python3
"""P0 ground truth for the probing experiments (RoboTwin2 ``batch_v1``).

Per episode (``<task>/<variant>/episode<N>/``) this derives, from the simulator
ground truth only (GT depth, robot / object masks, task-instance ids, end-effector
poses, gripper opening):

* manipulated-object 3D trajectory in the world frame (visible-surface median of
  the back-projected GT depth under the task-instance mask);
* interaction events: ``grasp_close`` / ``release_open`` (gripper crossings),
  ``motion_onset`` (object starts moving while a hand is near = contact
  establishment), ``liftoff`` (object leaves its resting height = support
  transfer), ``settle`` (object stops after moving = placement / release);
* per-frame token-grid maps (head camera, configurable stride): body / object /
  manipulated-object pixel fractions, body-object 3D min distance and the derived
  interaction zone (distance < ``--contact-m``);
* coupling field on the GT Track4D (``track4d/phase*.h5``): token-grid
  displacement, ``c``, ``dc``, ``pi`` and frame-level transition strength ``s_n``
  and motion magnitude ``m_n`` (uses ``metiswam4d.focus.coupling`` with a fixed
  Gaussian kernel; ``phi = |dc|``).

Outputs one ``.npz`` per episode plus ``index.jsonl``; ``summarize`` aggregates
window-level event statistics and role coverage (Experiments §3.2).

Usage::

    python scripts/probe/gt_events.py run --per-variant 10 --workers 64
    python scripts/probe/gt_events.py run --episodes list.txt
    python scripts/probe/gt_events.py summarize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes/gt_events")
REPO = Path(__file__).resolve().parents[2]

H, W = 240, 320
WINDOW = 33          # obs frame + 32 future frames (stride-4 -> 8 displacement frames)
STRIDE_T = 4

# ----------------------------------------------------------------------------- helpers


def unpack_bits(bits: np.ndarray, width: int = W) -> np.ndarray:
    return np.unpackbits(bits, axis=-1, bitorder="little")[..., :width].astype(bool)


def backproject(depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray, E: np.ndarray, grid: tuple[np.ndarray, np.ndarray]):
    """Pixels under ``mask`` -> world xyz using OpenCV intrinsics and world->cam ``E`` (3x4)."""
    ys, xs = grid
    z = depth_m[mask]
    x = (xs[mask] - K[0, 2]) * z / K[0, 0]
    y = (ys[mask] - K[1, 2]) * z / K[1, 1]
    p_cam = np.stack([x, y, z], 1)
    return (p_cam - E[:, 3]) @ E[:, :3]


def pool_fraction(mask: np.ndarray, stride: int) -> np.ndarray:
    """``[..., H, W]`` bool -> ``[..., H/stride, W/stride]`` fraction of set pixels."""
    h, w = mask.shape[-2] // stride, mask.shape[-1] // stride
    m = mask[..., : h * stride, : w * stride].reshape(*mask.shape[:-2], h, stride, w, stride)
    return m.mean(axis=(-1, -3), dtype=np.float32)


def pool_min(values: np.ndarray, stride: int) -> np.ndarray:
    h, w = values.shape[-2] // stride, values.shape[-1] // stride
    v = values[..., : h * stride, : w * stride].reshape(*values.shape[:-2], h, stride, w, stride)
    return v.min(axis=(-1, -3))


def pool_mean_valid(values: np.ndarray, valid: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """``values [N,H,W,C]``, ``valid [N,H,W]`` -> mean over valid pixels per token, valid fraction."""
    n, hh, ww, c = values.shape
    h, w = hh // stride, ww // stride
    v = values[:, : h * stride, : w * stride].reshape(n, h, stride, w, stride, c)
    m = valid[:, : h * stride, : w * stride].reshape(n, h, stride, w, stride, 1).astype(np.float32)
    s = (v * m).sum(axis=(2, 4))
    cnt = m.sum(axis=(2, 4))
    return s / np.maximum(cnt, 1.0), (cnt[..., 0] / (stride * stride)).astype(np.float32)


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.nanmedian(xp[i : i + k], axis=0)
    return out


def crossings(sig: np.ndarray, thr: float, direction: str) -> list[int]:
    a, b = sig[:-1], sig[1:]
    if direction == "down":
        idx = np.where((a >= thr) & (b < thr))[0] + 1
    else:
        idx = np.where((a <= thr) & (b > thr))[0] + 1
    return [int(i) for i in idx]


# ----------------------------------------------------------------------------- events


def segments(flag: np.ndarray, gap: int) -> list[list[int]]:
    """Contiguous True runs of ``flag``; runs separated by <= ``gap`` False frames are merged."""
    out: list[list[int]] = []
    t, T = 0, len(flag)
    while t < T:
        if flag[t]:
            s = t
            while t < T and flag[t]:
                t += 1
            if out and s - out[-1][1] <= gap:
                out[-1][1] = t - 1
            else:
                out.append([s, t - 1])
        else:
            t += 1
    return out


def detect_events(moving_cnt: np.ndarray, moving_minz: np.ndarray, moving_xyz: np.ndarray, contact_frame: np.ndarray,
                  ee_xyz: np.ndarray, gripper: np.ndarray, *, min_pix: int, lift_m: float, stop_frames: int) -> list[dict]:
    """Frame-level, instance-free events.

    ``moving_cnt [T]``   number of object pixels whose surface moved (see ``process_episode``)
    ``moving_minz [T]``  5th-percentile world z of the moving object points (nan if none)
    ``moving_xyz [T,3]`` median world xyz of the moving object points (nan if none)
    ``contact_frame [T]`` True when the body-object 3D min distance is below the contact threshold
    ``gripper [T,2]``    1 open, 0 closed
    """
    T = len(moving_cnt)
    ev: list[dict] = []
    for arm in range(2):
        for f in crossings(gripper[:, arm], 0.5, "down"):
            ev.append(dict(type="grasp_close", frame=f, arm=arm))
        for f in crossings(gripper[:, arm], 0.5, "up"):
            ev.append(dict(type="release_open", frame=f, arm=arm))
    # body-object contact episodes (covers pressing / pushing tasks where nothing is lifted)
    for s, e in segments(contact_frame, stop_frames):
        if e - s + 1 >= 2:
            ev.append(dict(type="contact_start", frame=int(s), arm=-1))
            if e < T - 2:
                ev.append(dict(type="contact_end", frame=int(e), arm=-1))
    moving = moving_cnt >= min_pix
    for s, e in segments(moving, stop_frames):
        if e - s + 1 < 3:
            continue
        xyz = moving_xyz[s : e + 1]
        ok = ~np.isnan(xyz[:, 0])
        if ok.sum() < 2:
            continue
        # interaction = a hand is in contact (3D) around the onset; otherwise it is e.g. an object falling
        if not contact_frame[max(s - 3, 0) : s + 4].any():
            continue
        d = np.linalg.norm(ee_xyz[s] - np.nanmedian(xyz[:3], axis=0), axis=-1)
        arm = int(np.argmin(d))
        ev.append(dict(type="motion_onset", frame=int(s), arm=arm))
        z0 = np.nanmedian(moving_minz[s : min(s + 3, e + 1)])
        lifted = np.where(np.nan_to_num(moving_minz[s : e + 1], nan=z0) - z0 > lift_m)[0]
        if len(lifted):
            ev.append(dict(type="liftoff", frame=int(s + lifted[0]), arm=arm))
        if e < T - 2:
            ev.append(dict(type="settle", frame=int(e), arm=arm))
    ev.sort(key=lambda d: d["frame"])
    return ev


# ----------------------------------------------------------------------------- per-episode


def process_episode(ep_dir: str, out_dir: str, stride: int, contact_m: float, move_mm: float, lift_cm: float,
                    stop_frames: int, min_pix: int, lag: int) -> dict:
    import h5py
    import torch
    from scipy import ndimage
    from scipy.spatial import cKDTree

    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO))
    from metiswam4d.focus.coupling import SpatialKernel, compute_coupling_field

    t_start = time.time()
    ep = Path(ep_dir)
    task, variant, name = ep.parts[-3], ep.parts[-2], ep.parts[-1]
    key = f"{task}/{variant}/{name}"
    out_path = Path(out_dir) / task / variant / f"{name}.npz"
    if out_path.exists():
        return dict(key=key, status="exists", path=str(out_path))
    for f in ("source.hdf5", "masks.h5", "track4d/phase0.h5"):
        if not (ep / f).exists():
            return dict(key=key, status="missing", detail=f)

    with h5py.File(ep / "source.hdf5", "r") as g:
        cam = g["observation/head_camera"]
        depth = cam["depth"][:].astype(np.float32) / 1000.0               # mm -> m
        K = cam["intrinsic_cv"][:]
        E = cam["extrinsic_cv"][:]
        ee = np.stack([g["endpose/left_endpose"][:, :3], g["endpose/right_endpose"][:, :3]], 1)
        grip = np.stack([g["endpose/left_gripper"][:], g["endpose/right_gripper"][:]], 1).astype(np.float32)
    T = depth.shape[0]
    with h5py.File(ep / "masks.h5", "r") as m:
        bits = m["head_camera/mask_bits"][:]
    body = unpack_bits(bits[:, 0])
    objs = unpack_bits(bits[:, 1])
    grid = np.mgrid[0:H, 0:W]
    grid = (grid[0], grid[1])
    valid_d = depth > 0.05
    body &= valid_d
    objs &= valid_d

    # world points of every object / body pixel, per frame (camera is static, but K/E are per frame anyway)
    pts_obj, pts_body = [], []
    for t in range(T):
        Kt, Et = K[t] if K.ndim == 3 else K, E[t] if E.ndim == 3 else E
        pts_obj.append(backproject(depth[t], objs[t], Kt, Et, grid))
        pts_body.append(backproject(depth[t], body[t], Kt, Et, grid))

    # --- body-object 3D min distance per pixel -> interaction zone
    min_dist = np.full((T, H, W), np.inf, dtype=np.float32)
    for t in range(T):
        if len(pts_body[t]) and len(pts_obj[t]):
            d_o, _ = cKDTree(pts_body[t][::2]).query(pts_obj[t], k=1, distance_upper_bound=0.15)
            d_b, _ = cKDTree(pts_obj[t][::2]).query(pts_body[t], k=1, distance_upper_bound=0.15)
            md = min_dist[t]
            md[objs[t]] = d_o
            md[body[t]] = np.minimum(md[body[t]], d_b)
    contact_frame = (min_dist.reshape(T, -1).min(axis=1) < contact_m)

    # --- instance-free object motion: an object pixel at t "moved" if its 3D point has no object surface
    #     within ``move_mm * lag`` at t+lag and it is not simply occluded by the robot at t+lag
    thr = move_mm / 1000.0 * lag
    moving_pix = np.zeros((T, H, W), dtype=bool)
    for t in range(T - lag):
        if len(pts_obj[t]) == 0 or len(pts_obj[t + lag]) == 0:
            continue
        d, _ = cKDTree(pts_obj[t + lag]).query(pts_obj[t], k=1, distance_upper_bound=0.15)
        cand = np.zeros((H, W), dtype=bool)
        cand[objs[t]] = d > thr
        cand &= ~body[t + lag]                      # newly occluded by the arm, not motion
        moving_pix[t] = cand
    # remove isolated speckles
    moving_pix = ndimage.binary_opening(moving_pix, structure=np.ones((1, 3, 3), dtype=bool))
    moving_cnt = moving_pix.reshape(T, -1).sum(axis=1)
    moving_minz = np.full(T, np.nan, dtype=np.float32)
    moving_xyz = np.full((T, 3), np.nan, dtype=np.float32)
    for t in range(T):
        if moving_cnt[t] >= min_pix:
            Kt, Et = K[t] if K.ndim == 3 else K, E[t] if E.ndim == 3 else E
            p = backproject(depth[t], moving_pix[t], Kt, Et, grid)
            moving_minz[t] = np.percentile(p[:, 2], 5)
            moving_xyz[t] = np.median(p, axis=0)

    # manipulated region = 2D object component(s) containing moving pixels (dilated in time by ``stop_frames``)
    manip = np.zeros((T, H, W), dtype=bool)
    last_hit = -10**6
    for t in range(T):
        if moving_cnt[t] >= min_pix:
            lab, n = ndimage.label(objs[t])
            hit = np.unique(lab[moving_pix[t]])
            manip[t] = np.isin(lab, hit[hit > 0])
            last_hit = t
    # propagate the manipulated component to frames without motion (nearest moving frame, same 2D overlap)
    mv_frames = np.where(moving_cnt >= min_pix)[0]
    if len(mv_frames):
        for t in range(T):
            if moving_cnt[t] < min_pix:
                ref = mv_frames[np.argmin(np.abs(mv_frames - t))]
                lab, n = ndimage.label(objs[t])
                hit = np.unique(lab[manip[ref] & objs[t]])
                manip[t] = np.isin(lab, hit[hit > 0])

    events = detect_events(moving_cnt, moving_minz, moving_xyz, contact_frame, ee, grip,
                           min_pix=min_pix, lift_m=lift_cm / 100.0, stop_frames=stop_frames)
    entities = {}
    ti_path = ep / "task_instances.h5"
    if ti_path.exists():
        with h5py.File(ti_path, "r") as ti:
            if "task_entities_json" in ti.attrs:
                entities = json.loads(ti.attrs["task_entities_json"])

    role_body = pool_fraction(body, stride)
    role_obj = pool_fraction(objs, stride)
    role_manip = pool_fraction(manip, stride)
    tok_min_dist = pool_min(min_dist, stride)
    interaction = tok_min_dist < contact_m

    # coupling field on GT Track4D, per phase
    kernel = SpatialKernel(size=5, sigma=1.0, mode="gaussian")
    coupling: dict[str, np.ndarray] = {}
    for phase in range(STRIDE_T):
        p = ep / "track4d" / f"phase{phase}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as th:
            disp = th["delta_xyz_cam"][:].astype(np.float32)             # [N,H,W,3] metres, source cam frame
            valid = unpack_bits(th["valid_bits"][:])
            src = th["source_frame_index"][:]
        d_tok, valid_frac = pool_mean_valid(disp, valid, stride)      # [N,h,w,3], [N,h,w]
        rb = role_body[src]
        ro = role_obj[src]
        bg = np.clip(1.0 - rb - ro, 0.0, 1.0)
        role = np.stack([bg, rb, ro], -1)
        with torch.no_grad():
            field = compute_coupling_field(torch.from_numpy(d_tok)[None], torch.from_numpy(role)[None], kernel)
        c = field.coupling[0].numpy()
        dc = field.transition[0].numpy()
        pi = field.proximity[0].numpy()
        mag = np.linalg.norm(d_tok, axis=-1)
        coupling[f"p{phase}_src"] = src.astype(np.int32)
        coupling[f"p{phase}_disp"] = d_tok.astype(np.float16)
        coupling[f"p{phase}_valid"] = valid_frac.astype(np.float16)
        coupling[f"p{phase}_c"] = c.astype(np.float16)
        coupling[f"p{phase}_dc"] = dc.astype(np.float16)
        coupling[f"p{phase}_pi"] = pi.astype(np.float16)
        coupling[f"p{phase}_s_n"] = (ro * np.abs(dc)).sum(axis=(1, 2)).astype(np.float32)   # phi = |dc|
        coupling[f"p{phase}_m_n"] = (ro * mag).sum(axis=(1, 2)).astype(np.float32)          # motion magnitude
        coupling[f"p{phase}_body_speed"] = (rb * mag).sum(axis=(1, 2)).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        key=key, frames=T, stride=stride, contact_m=contact_m,
        entities=json.dumps(entities), ee_xyz=ee.astype(np.float32), gripper=grip,
        moving_cnt=moving_cnt.astype(np.int32), moving_minz=moving_minz, moving_xyz=moving_xyz,
        contact_frame=contact_frame, frame_min_dist=min_dist.reshape(T, -1).min(axis=1).astype(np.float16),
        moving_pix=np.packbits(moving_pix, axis=-1, bitorder="little"),
        events=json.dumps(events),
        role_body=role_body.astype(np.float16), role_obj=role_obj.astype(np.float16),
        role_manip=role_manip.astype(np.float16), tok_min_dist=tok_min_dist.astype(np.float16),
        interaction=np.packbits(interaction, axis=-1, bitorder="little"),
        interaction_pix=np.packbits(min_dist < contact_m, axis=-1, bitorder="little"),
        **coupling,
    )
    return dict(key=key, status="ok", path=str(out_path), frames=T,
                moving_frames=int((moving_cnt >= min_pix).sum()), n_events=len(events), events=[f"{e['type']}@{e['frame']}" for e in events],
                seconds=round(time.time() - t_start, 1))


# ----------------------------------------------------------------------------- driver


def list_episodes(root: Path, per_variant: int | None, tasks: list[str] | None) -> list[Path]:
    eps: list[Path] = []
    for task in sorted(p for p in root.iterdir() if p.is_dir()):
        if tasks and task.name not in tasks:
            continue
        for variant in sorted(p for p in task.iterdir() if p.is_dir() and p.name.startswith("demo_")):
            names = sorted((p for p in variant.iterdir() if p.name.startswith("episode")),
                           key=lambda p: int(p.name[7:]))
            if per_variant:
                names = names[:per_variant]
            eps.extend(names)
    return eps


def cmd_run(a: argparse.Namespace) -> None:
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.episodes:
        eps = [root / l.strip() for l in open(a.episodes) if l.strip()]
    else:
        eps = list_episodes(root, a.per_variant, a.tasks.split(",") if a.tasks else None)
    print(f"{len(eps)} episodes -> {out}", flush=True)
    index = open(out / "index.jsonl", "a")
    t0, done, fail = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_episode, str(e), str(out), a.stride, a.contact_m,
                          a.move_mm, a.lift_cm, a.stop_frames, a.min_pix, a.lag): e for e in eps}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception:
                r = dict(key=str(futs[fut].relative_to(root)), status="error", detail=traceback.format_exc()[-800:])
            if r["status"] in ("error", "missing"):
                fail += 1
            done += 1
            index.write(json.dumps(r) + "\n")
            index.flush()
            if done % 50 == 0 or done == len(eps):
                print(f"[{done}/{len(eps)}] fail={fail} {time.time() - t0:.0f}s", flush=True)


def cmd_summarize(a: argparse.Namespace) -> None:
    out = Path(a.out)
    files = sorted(out.glob("*/*/episode*.npz"))
    print(f"{len(files)} episodes")
    per_task: dict[str, dict] = {}
    ev_types = ["grasp_close", "release_open", "contact_start", "contact_end", "motion_onset", "liftoff", "settle"]
    key_types = {"contact_start", "motion_onset", "liftoff", "settle"}
    win_counts = np.zeros(4, dtype=np.int64)           # 0,1,2,3+ key events per window
    pos_hist = np.zeros(8, dtype=np.int64)             # event position within window (8 stride-4 bins)
    cov = dict(body=[], obj=[], manip=[], inter=[], inter_when_obj=[])
    for f in files:
        z = np.load(f, allow_pickle=False)
        task = str(z["key"]).split("/")[0]
        T = int(z["frames"])
        events = json.loads(str(z["events"]))
        d = per_task.setdefault(task, dict(episodes=0, frames=0, **{k: 0 for k in ev_types}, windows=0, win_multi=0, no_key_event=0))
        d["episodes"] += 1
        d["frames"] += T
        for e in events:
            d[e["type"]] += 1
        kf = np.array([e["frame"] for e in events if e["type"] in key_types], dtype=np.int64)
        if len(kf) == 0:
            d["no_key_event"] += 1
        for o in range(0, max(T - WINDOW + 1, 1), STRIDE_T):
            inside = kf[(kf > o) & (kf <= o + WINDOW - 1)]
            n = min(len(inside), 3)
            win_counts[n] += 1
            d["windows"] += 1
            d["win_multi"] += int(len(inside) >= 2)
            for e in inside:
                pos_hist[min((e - o - 1) // STRIDE_T, 7)] += 1
        rb, ro, rm = z["role_body"].astype(np.float32), z["role_obj"].astype(np.float32), z["role_manip"].astype(np.float32)
        inter = np.unpackbits(z["interaction"], axis=-1, bitorder="little")[..., : rb.shape[-1]].astype(bool)
        cov["body"].append((rb > 0).mean())
        cov["obj"].append((ro > 0).mean())
        cov["manip"].append((rm > 0).mean())
        cov["inter"].append(inter.mean())
        cov["inter_when_obj"].append(inter.sum() / max((ro > 0).sum(), 1))
    tot = win_counts.sum()
    summary = dict(
        episodes=len(files),
        window=dict(frames=WINDOW, stride=STRIDE_T, total=int(tot),
                    key_events_per_window={"0": float(win_counts[0] / tot), "1": float(win_counts[1] / tot),
                                           "2": float(win_counts[2] / tot), "3+": float(win_counts[3] / tot)},
                    event_position_hist=(pos_hist / max(pos_hist.sum(), 1)).round(4).tolist()),
        role_coverage_token_fraction={k: float(np.mean(v)) for k, v in cov.items()},
        per_task=per_task,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", default=str(ROOT))
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--per-variant", type=int, default=10)
    r.add_argument("--tasks", default="")
    r.add_argument("--episodes", default="", help="file with task/variant/episodeN lines")
    r.add_argument("--workers", type=int, default=48)
    r.add_argument("--stride", type=int, default=16, help="token grid stride in pixels")
    r.add_argument("--contact-m", type=float, default=0.03)
    r.add_argument("--move-mm", type=float, default=2.0, help="object surface speed threshold, mm/frame")
    r.add_argument("--lag", type=int, default=2, help="frame lag for the surface motion test")
    r.add_argument("--min-pix", type=int, default=40, help="moving object pixels needed to call a frame 'moving'")
    r.add_argument("--lift-cm", type=float, default=1.5)
    r.add_argument("--stop-frames", type=int, default=8)
    s = sub.add_parser("summarize")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
