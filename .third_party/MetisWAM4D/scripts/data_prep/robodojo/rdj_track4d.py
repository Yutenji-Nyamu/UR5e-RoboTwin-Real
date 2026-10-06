"""RDJ_MetisWAM4D: robot-only pixel-frame Track4D for RoboDojo (ARX X5) from released joint states.

Per episode this produces, under ``<OUT>/<task>/train_4d/episodeN/``:

    source.hdf5   -> symlink to GeoRobodojo_janustrack4d ``data/episodeN.hdf5`` (RGB x3 views, FK-rendered
                     robot depth + link ids of the head camera, cameras, joint states)
    raw.hdf5      -> symlink to the official RoboDojo episode (end-effector poses, gripper)
    track4d.h5    stride-4 pixel-frame displacement of every robot / gripper pixel of the head camera
                  (du px, dv px, dd m; float16), role map (0 unknown, 1 robot), EEF20 + joint vector
    meta.json     FK reproduction check, displacement histograms, EEF ranges

Geometry: the released replay rendered depth + link index for robot pixels only; the per-link rigid
motion between frames t and t+4 comes from URDF forward kinematics of the released joint states, so no
simulator is needed here.  Non-robot pixels are *unknown* (objects were never reconstructed), never
"static background".  EEF20 = per arm [xyz (world), rot6d, gripper], from the official ``state/*_ee_poses``
(w, x, y, z quaternion = link6 pose, verified against FK to 0.1 mm) and ``state/*_ee_joint_states``.

Sub-commands
    build     process a shard of the manifest (resumable, multi-process)               [CPU]
    finalize  index.jsonl (+ split), eef20_stats.json, uvd_stats.json, qc.jsonl, root symlinks
    overlay   FK robot mask contour over the real RGB for a few frames per task -> _qc/overlays/  (alignment check)
    status    count complete / missing / failed episodes
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
import socket
import time
import traceback
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

JANUS = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobodojo_janustrack4d")
IMPERFECT = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRoboDojo_JanusAct4D_Imperfect")
URDF = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo/Assets/Robots/x5/X5A.urdf")
OUT = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D")
VARIANT = "train_4d"
STRIDE = 4
VAL_EPISODES_PER_TASK = 3          # highest episode numbers of every task are held out
FK_TOLERANCE_M = 1e-4              # p99 |FK(t->t+1) - released stride-1 track|
SCHEMA = "metiswam4d.rdj_track4d.uvd.v1"

# Histogram bins for the displacement statistics (robot pixels): |du|,|dv| in px, |dd| in m.
UV_BINS = np.arange(0.0, 120.25, 0.25)
D_BINS = np.arange(0.0, 0.4005, 0.0005)


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------


def read_manifest() -> list[dict]:
    with open(JANUS / "robodojo_fusion_v3_manifest.tsv") as f:
        rows = [{"task": r["task"], "episode": int(r["episode"])} for r in csv.DictReader(f, delimiter="\t")]
    return sorted(rows, key=lambda r: (r["task"], r["episode"]))


def episode_dir(task: str, episode: int) -> Path:
    return OUT / task / VARIANT / f"episode{episode}"


def is_complete(dest: Path) -> bool:
    return (dest / "meta.json").exists() and (dest / "track4d.h5").exists()


# ----------------------------------------------------------------------------
# forward kinematics (ARX X5 URDF, released joint states) -- mirrors the replay that rendered the geometry
# ----------------------------------------------------------------------------

ROOT_XY = {"left": -0.3, "right": 0.3}
ROOT_Y, ROOT_Z = -0.45, 0.765
GRIPPER_MIN_M, GRIPPER_MAX_M = -0.01, 0.044


def urdf_joints(path: Path) -> list[dict]:
    joints = []
    for j in ET.parse(path).getroot().findall("joint"):
        origin = j.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        t = np.eye(4)
        t[:3, 3] = xyz
        t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        kind = j.get("type")
        item = {"parent": j.find("parent").get("link"), "child": j.find("child").get("link"), "origin": t, "type": kind}
        if kind != "fixed":
            item["axis"] = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            item["index"] = int(j.get("name").replace("joint", "")) - 1   # joint1..6 arm, joint7/8 gripper
        joints.append(item)
    return joints


def forward_kinematics(joints: list[dict], arm: np.ndarray, gripper: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """``arm [n, 6]`` radians, ``gripper [n]`` normalised 0..1 -> {link: [n, 4, 4] world pose}."""
    n = len(arm)
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    root[:3, 3] = [ROOT_XY[side], ROOT_Y, ROOT_Z]
    poses = {"base_link": np.broadcast_to(root, (n, 4, 4)).copy()}
    grip = GRIPPER_MIN_M + np.clip(gripper, 0.0, 1.0) * (GRIPPER_MAX_M - GRIPPER_MIN_M)
    for j in joints:
        motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        if j["type"] != "fixed":
            value = arm[:, j["index"]] if j["index"] < 6 else grip
            if j["type"] == "prismatic":
                motion[:, :3, 3] = value[:, None] * j["axis"]
            else:
                motion[:, :3, :3] = Rotation.from_rotvec(value[:, None] * j["axis"]).as_matrix()
        poses[j["child"]] = poses[j["parent"]] @ j["origin"] @ motion
    return {f"{side}/{name}": p for name, p in poses.items()}


def joint_states(src: h5py.File, raw_path: Path) -> dict[str, np.ndarray]:
    """``{left_arm [n,6], left_gripper [n], right_arm, right_gripper, vector [n,14]}``; a few replay files lack the
    ``joint_state`` group, then the released joint states are read from the official episode."""
    if "joint_state" in src:
        return {k: src[f"joint_state/{k}"][:].astype(np.float64) for k in ("left_arm", "left_gripper", "right_arm", "right_gripper", "vector")}
    with h5py.File(raw_path) as raw:
        out = {}
        for side in ("left", "right"):
            out[f"{side}_arm"] = raw[f"state/{side}_arm_joint_states"][:].astype(np.float64)
            out[f"{side}_gripper"] = raw[f"state/{side}_ee_joint_states"][:, 0].astype(np.float64)
    out["vector"] = np.concatenate((out["left_arm"], out["left_gripper"][:, None], out["right_arm"], out["right_gripper"][:, None]), axis=1)
    return out


def link_poses(states: dict[str, np.ndarray], joints: list[dict]) -> dict[str, np.ndarray]:
    poses = {}
    for side in ("left", "right"):
        poses.update(forward_kinematics(joints, states[f"{side}_arm"], states[f"{side}_gripper"], side))
    return poses


def rigid_delta(points_cam: np.ndarray, ids: np.ndarray, valid: np.ndarray, id_to_link: dict[int, str],
                poses: dict[str, np.ndarray], world_to_cam: np.ndarray, t: int, u: int) -> np.ndarray:
    """Per-pixel camera-frame displacement ``[H, W, 3]`` of robot pixels from frame ``t`` to ``u``."""
    delta = np.zeros_like(points_cam)
    for index in np.unique(ids[valid]):
        pose = poses[id_to_link[int(index)]]
        transform = world_to_cam @ pose[u] @ np.linalg.inv(pose[t]) @ np.linalg.inv(world_to_cam)
        sel = valid & (ids == index)
        p = points_cam[sel]
        delta[sel] = p @ transform[:3, :3].T + transform[:3, 3] - p
    return delta


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[..., 2:3], 1e-6, None)
    return np.concatenate((points_cam[..., 0:1] / z * K[0, 0] + K[0, 2], points_cam[..., 1:2] / z * K[1, 1] + K[1, 2]), -1)


# ----------------------------------------------------------------------------
# end effector
# ----------------------------------------------------------------------------


def read_eef20(raw: h5py.File) -> tuple[np.ndarray, dict]:
    """Official ``state/*_ee_poses`` (xyz + wxyz quaternion, link6 in world) + gripper -> ``[n, 20]``."""
    arms, info = [], {}
    for side in ("left", "right"):
        pose = raw[f"state/{side}_ee_poses"][:].astype(np.float64)
        grip = raw[f"state/{side}_ee_joint_states"][:, 0].astype(np.float64)
        quat_xyzw = pose[:, [4, 5, 6, 3]]
        mat = Rotation.from_quat(quat_xyzw).as_matrix()
        rot6d = np.concatenate((mat[:, :, 0], mat[:, :, 1]), axis=-1)
        arms.append(np.concatenate((pose[:, :3], rot6d, grip[:, None]), axis=-1))
        act = raw[f"action/{side}_ee_poses"][:].astype(np.float64)
        info[f"{side}_action_minus_next_state_max"] = float(np.abs(act[:-1] - pose[1:]).max())
        info[f"{side}_rot6d_max_step"] = float(np.abs(np.diff(rot6d, axis=0)).max()) if len(rot6d) > 1 else 0.0
    return np.concatenate(arms, axis=-1).astype(np.float32), info


# ----------------------------------------------------------------------------
# per-episode build
# ----------------------------------------------------------------------------


def build_episode(row: dict, joints: list[dict]) -> dict:
    task, episode = row["task"], row["episode"]
    dest = episode_dir(task, episode)
    if is_complete(dest):
        return {"task": task, "episode": episode, "status": "exists"}
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    src_path = JANUS / task / VARIANT / "data" / f"episode{episode}.hdf5"
    old_path = JANUS / task / VARIANT / "track_head_camera" / f"episode{episode}.hdf5"

    with h5py.File(src_path) as src, h5py.File(old_path) as old:
        raw_path = Path(src["metadata/source_episode"][()].decode())
        head = src["observation/head_camera"]
        n, h, w = head["depth"].shape
        K = head["intrinsic_cv"][:].astype(np.float64)
        E = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        E[:, :3] = head["extrinsic_cv"][:]
        if np.abs(K - K[0]).max() > 1e-6 or np.abs(E - E[0]).max() > 1e-6:
            raise ValueError("head camera intrinsics / extrinsics vary within the episode")
        K, E = K[0], E[0]
        replay_timing = src["metadata/replay_timing"][()].decode()
        id_to_link = {int(k): v for k, v in json.loads(old.attrs["link_index_json"]).items()}
        states = joint_states(src, raw_path)
        poses = link_poses(states, joints)
        qpos = states["vector"].astype(np.float32)

        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        rays = np.stack(((xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)), -1)

        n_old = old["link_index"].shape[0]   # released stride-1 track has n - 1 rows

        def frame_geometry(t: int):
            depth = head["depth"][t].astype(np.float64) * 1e-3
            ids = old["link_index"][t] if t < n_old else None
            valid = (depth > 0) if ids is None else (ids > 0) & (depth > 0)
            return rays * depth[..., None], ids, valid

        # FK reproduction against the released stride-1 track (catches mistimed replays such as stack_bowls).
        fk_check = []
        for t in (0, n // 3, 2 * n // 3, n - 2):
            pts, ids, valid = frame_geometry(t)
            delta = rigid_delta(pts, ids, valid, id_to_link, poses, E, t, t + 1)
            err = np.abs(delta - old["delta_xyz_cam"][t].astype(np.float64))[valid]
            fk_check.append({"frame": int(t), "p99_m": float(np.quantile(err, 0.99)) if err.size else 0.0,
                             "max_m": float(err.max()) if err.size else 0.0,
                             "valid_match": bool(np.array_equal(valid, old["valid"][t] > 0)),
                             "depth_only_mask_match": bool(np.array_equal(valid, pts[..., 2] > 0))})
        fk_ok = max(c["p99_m"] for c in fk_check) <= FK_TOLERANCE_M
        if not fk_ok:
            meta = {"task": task, "episode": episode, "frames": int(n), "status": "fk_mismatch", "replay_timing": replay_timing,
                    "fk_check": fk_check, "seconds": round(time.time() - started, 1)}
            (dest / "meta.failed.json").write_text(json.dumps(meta, indent=1))
            return {"task": task, "episode": episode, "status": "fk_mismatch"}

        with h5py.File(raw_path) as raw:
            eef20, eef_info = read_eef20(raw)
        if len(eef20) != n or len(qpos) != n:
            raise ValueError(f"length mismatch: frames {n}, eef {len(eef20)}, qpos {len(qpos)}")

        hist_u = np.zeros(len(UV_BINS) - 1, np.int64)
        hist_v = np.zeros(len(UV_BINS) - 1, np.int64)
        hist_d = np.zeros(len(D_BINS) - 1, np.int64)
        count, clipped_u, clipped_v, clipped_d, robot_pixels = 0, 0, 0, 0, 0
        tmp = dest / "track4d.tmp.h5"
        with h5py.File(tmp, "w") as out:
            out.attrs.update(schema=SCHEMA, stride=STRIDE, camera="head_camera", coordinate_frame="source_head_camera_cv",
                             units="du px, dv px, dd m", width=w, height=h, frames=n, role_labels="0 unknown, 1 robot",
                             replay_timing=replay_timing, source=str(src_path), raw=str(raw_path), urdf=str(URDF),
                             eef20_layout="left xyz(3) rot6d(6) gripper(1) | right xyz(3) rot6d(6) gripper(1); world frame, link6",
                             quaternion_order="wxyz (official state/*_ee_poses)")
            out.create_dataset("intrinsic_cv", data=K.astype(np.float32))
            out.create_dataset("extrinsic_cv", data=E[:3].astype(np.float32))
            out.create_dataset("eef20", data=eef20)
            out.create_dataset("qpos", data=qpos)
            d_uvd = out.create_dataset("delta_uvd", (n - STRIDE, h, w, 3), dtype="f2", chunks=(1, h, w, 3),
                                       compression="lzf", shuffle=True)
            d_role = out.create_dataset("role", (n, h, w), dtype="u1", chunks=(1, h, w), compression="lzf")
            for t in range(n):
                pts, ids, valid = frame_geometry(t)
                d_role[t] = valid.astype(np.uint8)
                robot_pixels += int(valid.sum())
                if t >= n - STRIDE:
                    continue
                delta = rigid_delta(pts, ids, valid, id_to_link, poses, E, t, t + STRIDE)
                moved = pts + delta
                uv0, uv1 = project(pts, K), project(moved, K)
                uvd = np.concatenate((uv1 - uv0, delta[..., 2:3]), -1)
                uvd[~valid] = 0.0
                d_uvd[t] = uvd.astype(np.float16)
                v = uvd[valid]
                hist_u += np.histogram(np.abs(v[:, 0]), UV_BINS)[0]
                hist_v += np.histogram(np.abs(v[:, 1]), UV_BINS)[0]
                hist_d += np.histogram(np.abs(v[:, 2]), D_BINS)[0]
                count += len(v)
                clipped_u += int((np.abs(v[:, 0]) >= UV_BINS[-1]).sum())
                clipped_v += int((np.abs(v[:, 1]) >= UV_BINS[-1]).sum())
                clipped_d += int((np.abs(v[:, 2]) >= D_BINS[-1]).sum())
            out.attrs["complete"] = True
        os.replace(tmp, dest / "track4d.h5")

    for name, target in (("source.hdf5", src_path), ("raw.hdf5", raw_path)):
        link = dest / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)

    meta = {
        "task": task, "episode": episode, "frames": int(n), "status": "ok", "replay_timing": replay_timing,
        "robot_pixels_per_frame": robot_pixels / n, "fk_check": fk_check,
        "eef": {"min": eef20.min(0).tolist(), "max": eef20.max(0).tolist(), **eef_info},
        "uvd_hist": {"count": int(count), "u": hist_u.tolist(), "v": hist_v.tolist(), "d": hist_d.tolist(),
                     "beyond_last_bin": [clipped_u, clipped_v, clipped_d]},
        "seconds": round(time.time() - started, 1), "host": socket.gethostname(),
    }
    (dest / "meta.json").write_text(json.dumps(meta))
    return {"task": task, "episode": episode, "status": "ok", "frames": int(n), "seconds": meta["seconds"]}


def _worker(row: dict) -> dict:
    try:
        return build_episode(row, urdf_joints(URDF))
    except Exception as exc:  # noqa: BLE001 - recorded, the shard goes on
        dest = episode_dir(row["task"], row["episode"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "meta.failed.json").write_text(json.dumps({"task": row["task"], "episode": row["episode"], "status": "error",
                                                           "error": repr(exc), "traceback": traceback.format_exc()}, indent=1))
        return {"task": row["task"], "episode": row["episode"], "status": "error", "error": repr(exc)[:200]}


def cmd_build(args) -> None:
    rows = read_manifest()[args.shard::args.num_shards]
    if args.tasks:
        rows = [r for r in rows if r["task"] in set(args.tasks)]
    todo = [r for r in rows if not is_complete(episode_dir(r["task"], r["episode"]))]
    print(f"[build] shard {args.shard}/{args.num_shards}: {len(rows)} episodes, {len(todo)} to do, {args.workers} workers", flush=True)
    done = 0
    started = time.time()
    with Pool(args.workers, maxtasksperchild=8) as pool:
        for result in pool.imap_unordered(_worker, todo):
            done += 1
            print(f"[{done}/{len(todo)} {time.time() - started:7.0f}s] {result}", flush=True)
    print("[build] done", flush=True)


# ----------------------------------------------------------------------------
# finalize
# ----------------------------------------------------------------------------


def quantiles_from_hist(hist: np.ndarray, bins: np.ndarray, qs=(0.5, 0.9, 0.99, 0.995, 0.999)) -> dict:
    total = hist.sum()
    if total == 0:
        return {}
    cdf = np.cumsum(hist) / total
    return {f"p{q * 100:g}": float(bins[1:][np.searchsorted(cdf, q)]) if q <= cdf[-1] else float(bins[-1]) for q in qs}


def cmd_finalize(args) -> None:
    rows = read_manifest()
    index, qc, failed = [], [], []
    eef_min, eef_max = None, None
    hist_u = np.zeros(len(UV_BINS) - 1, np.int64)
    hist_v = np.zeros(len(UV_BINS) - 1, np.int64)
    hist_d = np.zeros(len(D_BINS) - 1, np.int64)
    count, beyond, frames_total = 0, np.zeros(3, np.int64), 0
    by_task: dict[str, list[int]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["episode"])
    for r in rows:
        dest = episode_dir(r["task"], r["episode"])
        if not (dest / "meta.json").exists():
            reason = json.loads((dest / "meta.failed.json").read_text())["status"] if (dest / "meta.failed.json").exists() else "missing"
            failed.append({**r, "status": reason})
            continue
        meta = json.loads((dest / "meta.json").read_text())
        val_cut = sorted(by_task[r["task"]])[-VAL_EPISODES_PER_TASK:]
        index.append({"task": r["task"], "variant": VARIANT, "episode": r["episode"], "frames": meta["frames"],
                      "split": "val" if r["episode"] in val_cut else "train"})
        qc.append({"task": r["task"], "episode": r["episode"], "frames": meta["frames"], "fk_p99_m": max(c["p99_m"] for c in meta["fk_check"]),
                   "robot_pixels_per_frame": meta["robot_pixels_per_frame"],
                   "eef_action_minus_next_state_max": max(meta["eef"]["left_action_minus_next_state_max"], meta["eef"]["right_action_minus_next_state_max"]),
                   "eef_rot6d_max_step": max(meta["eef"]["left_rot6d_max_step"], meta["eef"]["right_rot6d_max_step"])})
        frames_total += meta["frames"]
        if index[-1]["split"] == "train":
            lo, hi = np.asarray(meta["eef"]["min"]), np.asarray(meta["eef"]["max"])
            eef_min = lo if eef_min is None else np.minimum(eef_min, lo)
            eef_max = hi if eef_max is None else np.maximum(eef_max, hi)
        hist_u += np.asarray(meta["uvd_hist"]["u"]); hist_v += np.asarray(meta["uvd_hist"]["v"]); hist_d += np.asarray(meta["uvd_hist"]["d"])
        count += meta["uvd_hist"]["count"]; beyond += np.asarray(meta["uvd_hist"]["beyond_last_bin"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "index.jsonl", "w") as f:
        for row in index:
            f.write(json.dumps(row) + "\n")
    with open(OUT / "qc.jsonl", "w") as f:
        for row in qc:
            f.write(json.dumps(row) + "\n")
    (OUT / "failed.json").write_text(json.dumps(failed, indent=1))
    (OUT / "eef20_stats.json").write_text(json.dumps({
        "embodiment": "robodojo_arx_x5", "layout": "left xyz(3) rot6d(6) gripper(1) | right xyz(3) rot6d(6) gripper(1)",
        "frame": "world (robot bases at x = -/+0.3, y = -0.45, z = 0.765; EE = link6)", "quaternion_source": "state/*_ee_poses wxyz",
        "scope": f"train split, {sum(r['split'] == 'train' for r in index)} episodes", "normalisation": "min-max to [-1, 1]",
        "min": eef_min.tolist(), "max": eef_max.tolist()}, indent=1))
    (OUT / "uvd_stats.json").write_text(json.dumps({
        "scope": "robot pixels, all built episodes, stride-4 pairs", "count": int(count), "width_px": 320,
        "abs_du_px": quantiles_from_hist(hist_u, UV_BINS), "abs_dv_px": quantiles_from_hist(hist_v, UV_BINS),
        "abs_dd_m": quantiles_from_hist(hist_d, D_BINS), "beyond_last_bin": beyond.tolist(),
        "frac_beyond_codec_scale": {"u_v_scale_px": 320 / 6, "d_scale_m": 0.10,
                                    "u": float(hist_u[UV_BINS[1:] > 320 / 6].sum() / max(count, 1)),
                                    "v": float(hist_v[UV_BINS[1:] > 320 / 6].sum() / max(count, 1)),
                                    "d": float(hist_d[D_BINS[1:] > 0.10].sum() / max(count, 1))}}, indent=1))
    for name, target in (("episode_instructions_official.jsonl", IMPERFECT / "episode_instructions_official.jsonl"),
                         ("text_cache", IMPERFECT / "_pipeline" / "text_cache_alpha_rdj")):
        link = OUT / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    (OUT / "assets").mkdir(exist_ok=True)
    shutil.copy2(URDF, OUT / "assets" / URDF.name)
    n_val = sum(r["split"] == "val" for r in index)
    print(f"[finalize] {len(index)} episodes ({n_val} val), {frames_total} frames, {len(failed)} failed; robot pixels {count}")
    print(json.dumps(json.loads((OUT / "uvd_stats.json").read_text())["abs_du_px"]), json.dumps(json.loads((OUT / "uvd_stats.json").read_text())["abs_dd_m"]))


def cmd_overlay(args) -> None:
    """One PNG per task: 4 frames, FK robot-mask contour (green) + du/dv arrows (red) over the real head-camera RGB."""
    import cv2
    import io
    from PIL import Image

    out_dir = OUT / "_qc" / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(OUT / "index.jsonl")] if (OUT / "index.jsonl").exists() else \
        [{**r, "frames": None} for r in read_manifest()]
    per_task: dict[str, dict] = {}
    for r in rows:
        if is_complete(episode_dir(r["task"], r["episode"])):
            per_task.setdefault(r["task"], r)
    for task, r in sorted(per_task.items()):
        d = episode_dir(task, r["episode"])
        with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
            n = t4d["role"].shape[0]
            panels = []
            for t in np.linspace(0, n - STRIDE - 1, 4).astype(int):
                # source JPEGs are BGR-ordered (cv2.imencode of an RGB array), so PIL's "RGB" is already cv2 BGR
                bgr = np.asarray(Image.open(io.BytesIO(bytes(src["observation/head_camera/rgb"][t]))).convert("RGB"))
                mask = (t4d["role"][t] > 0).astype(np.uint8)
                uvd = t4d["delta_uvd"][t].astype(np.float32)
                img = bgr.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
                ys, xs = np.nonzero(mask)
                for y, x in zip(ys[::400], xs[::400]):
                    du, dv = uvd[y, x, :2]
                    if abs(du) + abs(dv) > 0.5:
                        cv2.arrowedLine(img, (int(x), int(y)), (int(round(x + du)), int(round(y + dv))), (0, 0, 255), 1, tipLength=0.3)
                cv2.putText(img, f"t={t}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                panels.append(img)
            cv2.imwrite(str(out_dir / f"{task}_episode{r['episode']}.png"), np.concatenate(panels, axis=1))
    print(f"[overlay] {len(per_task)} tasks -> {out_dir}")


def cmd_status(args) -> None:
    rows = read_manifest()
    ok = sum(is_complete(episode_dir(r["task"], r["episode"])) for r in rows)
    failed = [r for r in rows if (episode_dir(r["task"], r["episode"]) / "meta.failed.json").exists()
              and not is_complete(episode_dir(r["task"], r["episode"]))]
    print(f"{ok}/{len(rows)} complete, {len(failed)} failed: {sorted({r['task'] for r in failed})}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tasks", nargs="*", default=None)
    sub.add_parser("finalize")
    sub.add_parser("overlay")
    sub.add_parser("status")
    args = parser.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize, "overlay": cmd_overlay, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
