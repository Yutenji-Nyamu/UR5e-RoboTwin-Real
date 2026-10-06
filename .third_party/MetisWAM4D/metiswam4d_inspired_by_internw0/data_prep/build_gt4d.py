"""Teacher rollouts with head-camera ground truth -> training episodes with arm / gripper / object uvd Track.

Input: ``<record>/<variant>/layout_NNN.hdf5`` written by ``iw0_deploy`` with ``METIS_RECORD_4D=1`` (``gt_head``: Isaac
depth, instance ids, id -> prim table, per-frame poses of every rigid / articulated scene instance).

Per episode ``<out>/<task>/rollout_<tag>/episodeN/``:
    source.hdf5   as ``scripts/data_prep/robodojo/rdj_rollout_dataset.py`` (RGB JPEGs, SAPIEN robot-only depth / link
                  ids, joint states) + ``joint_action/vector`` (= next joint state, the official convention)
    track4d.h5    stride-4 transitions for every frame (the four phases t mod 4 together cover all frames):
                  role 1 arm / 3 gripper (URDF FK, link6-8 + wrist camera = gripper), 2 object (every scene instance
                  that moves > ``--move-thresh`` m in the episode; rigid motion from its ground-truth poses applied to
                  the Isaac depth point), 0 known static background (delta 0); schema ``...gt4d.v1``
    meta.json     outcome, instruction, object instances, QC (object pixel share, centroid reprojection error)

    PYTHONPATH=.:<JanusTrack4d_260824> CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.10 \
        metiswam4d_inspired_by_internw0/data_prep/build_gt4d.py build --record <dir> --out <root> --tag t1 --workers 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
ROLLOUT = PROJECT / "scripts/data_prep/robodojo/rdj_rollout_dataset.py"
SCHEMA = "metiswam4d_iw0.rdj_track4d.gt4d.v1"
STRIDE = 4
GRIPPER_LINKS = ("link6", "link7", "link8", "camera_base", "camera")


def _rollout():
    spec = importlib.util.spec_from_file_location("rdj_rollout_dataset", ROLLOUT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose_matrix(p7: np.ndarray) -> np.ndarray:
    """``[x y z qw qx qy qz]`` -> 4x4."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([p7[4], p7[5], p7[6], p7[3]]).as_matrix()
    m[:3, 3] = p7[:3]
    return m


def instance_of_prim(names: list[str], id_to_prim: dict[int, str]) -> dict[int, int]:
    """Isaac instance id -> index into ``names`` (the scene instance whose name is a path component of the prim)."""
    out = {}
    for iid, prim in id_to_prim.items():
        parts = prim.split("/")
        for k, name in enumerate(names):
            if name in parts:
                out[iid] = k
                break
    return out


def gt_track(dest: Path, rec_path: Path, move_thresh: float) -> dict:
    with h5py.File(rec_path) as rec:
        g = rec["gt_head"]
        depth = g["depth_mm"][:].astype(np.float64) * 1e-3                 # [n, 240, 320] Isaac z-depth (m)
        inst = g["instance"][:]
        poses = g["instance_poses"][:].astype(np.float64)                  # [n, K, 7]
        names = json.loads(g.attrs["instance_names"])
        id_to_prim = {int(k): v for k, v in json.loads(g.attrs["id_to_prim"]).items()}
        k_live = rec["observation/head_camera/intrinsic_live"][:].astype(np.float64)   # 640x480
    with h5py.File(dest / "track4d.h5") as t4d:
        K = t4d["intrinsic_cv"][:].astype(np.float64)                      # 320x240 (SAPIEN)
        E = np.eye(4)
        E[:3] = t4d["extrinsic_cv"][:].astype(np.float64)                  # world -> camera (cv)
        robot_uvd = t4d["delta_uvd"][:]
        robot_role = t4d["role"][:]
        attrs = dict(t4d.attrs)
        eef20, qpos = t4d["eef20"][:], t4d["qpos"][:]
    with h5py.File(dest / "source.hdf5") as src:
        links = src["observation/head_camera/instance_id"][:]
    n, h, w = depth.shape

    travel = np.linalg.norm(poses[:, :, :3] - poses[:1, :, :3], axis=-1).max(axis=0)   # [K]
    moving = travel > move_thresh
    iid_to_k = instance_of_prim(names, id_to_prim)
    lut = np.full(max(list(id_to_prim) + [0]) + 1, -1, np.int64)
    for iid, k in iid_to_k.items():
        lut[iid] = k
    kidx = np.where(inst < len(lut), lut[np.minimum(inst, len(lut) - 1)], -1)        # [n, h, w]

    # Rays of the [::2, ::2] samples of the 640x480 Isaac image (pixel centres 2j + 0.5 in live pixels).
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    rays = np.stack(((2 * xx + 0.5 - k_live[0, 2]) / k_live[0, 0], (2 * yy + 0.5 - k_live[1, 2]) / k_live[1, 1],
                     np.ones_like(xx)), -1)
    uv_here = np.stack(((rays[..., 0] * K[0, 0] + K[0, 2]), (rays[..., 1] * K[1, 1] + K[1, 2])), -1)
    c2w = np.linalg.inv(E)
    mats = np.stack([[pose_matrix(poses[t, k]) for k in range(poses.shape[1])] for t in range(n)]) if poses.shape[1] else None

    roles = np.zeros((n, h, w), np.uint8)
    uvd = np.zeros((n - STRIDE, h, w, 3), np.float32)
    robot = robot_role > 0
    obj_pixels, centroid_err = 0, []
    for t in range(n):
        obj = (~robot[t]) & (kidx[t] >= 0) & moving[np.maximum(kidx[t], 0)] & (depth[t] > 0)
        roles[t][obj] = 2
        roles[t][robot[t]] = 1
        obj_pixels += int(obj.sum())
        if t < n - STRIDE:
            uvd[t][robot[t]] = robot_uvd[t][robot[t]]
            if obj.any():
                pts = rays[obj] * depth[t][obj][:, None]
                pw = (c2w[:3, :3] @ pts.T).T + c2w[:3, 3]
                ks = kidx[t][obj]
                rel = np.einsum("nij,njk->nik", mats[t + STRIDE][ks], np.linalg.inv(mats[t][ks]))
                pw2 = np.einsum("nij,nj->ni", rel[:, :3, :3], pw) + rel[:, :3, 3]
                pc2 = (E[:3, :3] @ pw2.T).T + E[:3, 3]
                uv2 = np.stack((pc2[:, 0] / pc2[:, 2] * K[0, 0] + K[0, 2], pc2[:, 1] / pc2[:, 2] * K[1, 1] + K[1, 2]), -1)
                uvd[t][obj] = np.concatenate((uv2 - uv_here[obj], (pc2[:, 2] - pts[:, 2])[:, None]), -1)
        for k in np.flatnonzero(moving):
            m = (kidx[t] == k) & ~robot[t]
            if m.sum() > 20:
                c = poses[t, k, :3]
                pc = E[:3, :3] @ c + E[:3, 3]
                if pc[2] > 0:
                    proj = np.array([pc[0] / pc[2] * K[0, 0] + K[0, 2], pc[1] / pc[2] * K[1, 1] + K[1, 2]])
                    centroid_err.append(float(np.linalg.norm(proj - uv_here[m].mean(0))))
    roles[(roles == 1) & np.isin(links, gripper_link_ids(dest))] = 3

    tmp = dest / "track4d.gt.tmp.h5"
    with h5py.File(tmp, "w") as out:
        out.attrs.update({k: v for k, v in attrs.items() if k not in ("schema", "role_labels")})
        out.attrs.update(schema=SCHEMA, role_labels="0 static background, 1 arm, 2 object, 3 gripper",
                         objects=json.dumps([names[k] for k in np.flatnonzero(moving)]), move_thresh=move_thresh)
        out.create_dataset("intrinsic_cv", data=K.astype(np.float32))
        out.create_dataset("extrinsic_cv", data=E[:3].astype(np.float32))
        out.create_dataset("eef20", data=eef20)
        out.create_dataset("qpos", data=qpos)
        out.create_dataset("delta_uvd", data=uvd.astype(np.float16), chunks=(1, h, w, 3), compression="lzf", shuffle=True)
        out.create_dataset("role", data=roles, chunks=(1, h, w), compression="lzf")
        out.attrs["complete"] = True
    os.replace(tmp, dest / "track4d.h5")
    return {"objects": [names[k] for k in np.flatnonzero(moving)], "object_pixels_per_frame": obj_pixels / n,
            "centroid_reproj_px_median": float(np.median(centroid_err)) if centroid_err else None}


_GRIPPER_IDS: dict[str, np.ndarray] = {}


def gripper_link_ids(dest: Path) -> np.ndarray:
    """SAPIEN link ids (1-based, ``DualX5Renderer.link_names`` order) of the gripper links."""
    key = "ids"
    if key not in _GRIPPER_IDS:
        from metiswam4d.eval.rdj_observation import URDF
        from preprocess.robodojo.official_geometry import DualX5Renderer
        k = np.array([[144.0, 0, 160], [0, 144.0, 120], [0, 0, 1]])
        names = DualX5Renderer(URDF, {"head_camera": (k, (240, 320, 3))}).link_names
        _GRIPPER_IDS[key] = np.array([i + 1 for i, nm in enumerate(names) if nm.split("/")[-1] in GRIPPER_LINKS])
    return _GRIPPER_IDS[key]


def build_one(job) -> dict:
    path, out, tag, episode, move_thresh = job
    R = _rollout()
    sys.modules["rdj_rollout_dataset"] = R
    try:
        result = R._build(Path(path), Path(out), tag, int(episode))
        if result["status"] not in ("ok", "exists"):
            return result
        dest = R.episode_dir(Path(out), result["task"], tag, int(episode))
        meta = json.loads((dest / "meta.json").read_text())
        if meta.get("schema") == SCHEMA:
            return {**result, "status": "exists"}
        with h5py.File(dest / "source.hdf5", "a") as src:
            if "joint_action/vector" not in src:
                q = src["joint_state/vector"][:]
                src.create_dataset("joint_action/vector", data=np.concatenate((q[1:], q[-1:]), 0))
        qc = gt_track(dest, Path(path), move_thresh)
        meta.update(schema=SCHEMA, **qc)
        (dest / "meta.json").write_text(json.dumps(meta))
        return {**result, **qc, "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - recorded, the batch goes on
        return {"source": str(path), "status": "error", "error": repr(exc)[:300], "trace": traceback.format_exc()[-1500:]}


def cmd_build(args) -> None:
    out, record = Path(args.out), Path(args.record)
    tasks = set(args.tasks.split(",")) if args.tasks else None
    jobs, counter = [], {}
    for path in sorted(record.glob("*/layout_*.hdf5")):
        try:
            with h5py.File(path) as rec:
                a = dict(rec.attrs)
                if "gt_head" not in rec or not bool(a["success"]) or int(a["frames"]) < 37:
                    continue
        except OSError:          # still being written by the recorder
            continue
        task = str(a["task"])
        if tasks and task not in tasks:
            continue
        counter[task] = counter.get(task, 0) + 1
        # stable numbering while recordings keep arriving: episode = seed offset + official layout id
        jobs.append((str(path), str(out), args.tag, args.episode_offset + int(a["layout_id"]), args.move_thresh))
    print(f"{len(jobs)} successful 4D recordings: {counter}", flush=True)
    log = out / "_logs" / f"build_{args.tag}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, open(log, "a") as handle:
        for r in pool.imap_unordered(build_one, jobs):
            handle.write(json.dumps(r) + "\n")
            handle.flush()
            print(json.dumps({k: r.get(k) for k in ("status", "task", "episode", "objects", "object_pixels_per_frame",
                                                    "centroid_reproj_px_median", "error")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--record", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tag", default="t1")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--tasks")
    b.add_argument("--episode-offset", type=int, default=0)
    b.add_argument("--move-thresh", type=float, default=0.02)
    args = ap.parse_args()
    cmd_build(args)


if __name__ == "__main__":
    main()
