"""IG-10K simulation episodes replayed with simulator-exact geometry for MetisWAM4D.

The LeRobot release keeps only 224x224 zed2i RGB-D, joint states and joint actions.  Each episode is re-simulated in
the Imitator-Game ManiSkill env: the reset seed is recovered by matching the first 224 frame (the collector tries seeds
0, 1, 2, ... and skips failed plans, so seeds increase within a directory), the recorded joint actions are replayed
open-loop, and the zed2i camera is re-rendered at 1280x720 with depth and per-entity segmentation.  Every pixel belongs
to one rigid entity (robot link, actor or articulation link), so its 3-D motion is exact:

    X_{t+4} = T_b(t+4) T_b(t)^-1 X_t ,     X_t back-projected from depth, b = segmentation(u, v)

Output ``<out>/<env_dir>/episode<k>.h5`` (written to a temp name, renamed when complete):
    rgb                 [T] JPEG bytes (RGB), 512x288 (zed2i 1280x720 area-downsampled 2.5x)
    depth_mm            [T, 180, 320] float16  (zed2i z-depth sampled at the centres of a 4x4-pixel grid)
    part                [T, 180, 320] uint8  0 static / 1 arm / 2 gripper / 3 movable object
    role                [T, 180, 320] uint8  0 static / 1 robot (arm + gripper) / 2 object
    delta_uvd           [T-4, 180, 320, 3] float16  (du px, dv px on the 180x320 grid, dd m), 0 where role == 0
    qpos18 / actions16  [T, 18] / [T, 16]  the LeRobot columns; sim_qpos18 [T, 18] the replayed joint state
    entity_pose         [T, E, 7]  (xyz, wxyz) of every segmentation entity; attrs entities = names / kinds
    intrinsic_cv [3, 3] (180x320 grid)  extrinsic_cv [3, 4] (world -> camera, OpenCV)
    attrs: env_dir, level, task, episode, seed, seed_mse, seed_mse_second, replay_success, schema, stride, qc_*

Usage (after `source scripts/ig10k/ms_env.sh`):
    /usr/bin/python3.10 scripts/ig10k/sim_gt4d.py --dirs L0_TwoRobotPickAppleBasket-v1 --episodes 0 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

SIM_ROOT = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/IG-10K-Dataset/imitator_sim_v1_zed2i")
OUT_ROOT = Path("/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/sim")
SCHEMA = "metiswam4d.ig10k_sim_track4d.uvd.objects.v1"
SRC_W, SRC_H = 1280, 720
RGB_W, RGB_H = 512, 288
GRID_H, GRID_W = 180, 320
STRIDE = 4
FPS = 30
GRIPPER_KEYS = ("hand", "finger", "tcp", "camera")
L_ENV_VARS = {"L1": "MANI_SKILL_L1", "L2": "MANI_SKILL_L2", "L3": "MANI_SKILL_L3"}

ROWS = np.floor((np.arange(GRID_H) + 0.5) * SRC_H / GRID_H).astype(np.int64)
COLS = np.floor((np.arange(GRID_W) + 0.5) * SRC_W / GRID_W).astype(np.int64)


def set_level(level: str) -> None:
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

    for v in L_ENV_VARS.values():
        os.environ.pop(v, None)
    if level in L_ENV_VARS:
        os.environ[L_ENV_VARS[level]] = "1"
    L0_L3_utils.set_l1_enabled(level == "L1")
    L0_L3_utils.set_l2_enabled(level == "L2")
    L0_L3_utils.set_l3_enabled(level == "L3")
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)


def to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def load_dir(env_dir: Path):
    data = pq.read_table(sorted((env_dir / "data").glob("chunk-*/*.parquet"))[0]).to_pandas()
    meta = pq.read_table(sorted((env_dir / "meta" / "episodes").glob("chunk-*/*.parquet"))[0]).to_pandas()
    return data, meta


def first_frames(env_dir: Path, meta, episodes: list[int]) -> dict[int, np.ndarray]:
    """First 224 zed2i frame of each requested episode, decoding the shared video file once."""
    path = env_dir / "videos" / "observation.images.zed2i" / "chunk-000" / "file-000.mp4"
    want = {}
    for ep in episodes:
        row = meta[meta["episode_index"] == ep].iloc[0]
        want[int(round(float(row["videos/observation.images.zed2i/from_timestamp"]) * FPS))] = ep
    out, last = {}, max(want)
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[want[i]] = f.to_ndarray(format="rgb24").astype(np.float32)
            if i >= last:
                break
    return out


def make_env(env_id: str, hi: bool, render_device: str):
    import gymnasium as gym

    cams = dict(shader_pack="rt-fast")
    if hi:
        cams["zed2i"] = dict(width=SRC_W, height=SRC_H)
    return gym.make(env_id, num_envs=1, obs_mode="rgb+depth+segmentation" if hi else "rgb",
                    control_mode="pd_joint_pos", render_mode="rgb_array", sim_backend="physx_cpu",
                    render_backend=render_device, sensor_configs=cams, max_episode_steps=4000)


def first_frame_mse(env, ref: np.ndarray, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    img = to_np(obs["sensor_data"]["zed2i"]["rgb"])[0].astype(np.float32)
    return float(((img - ref) ** 2).mean())


def split_action(a16: np.ndarray, keys: list[str]) -> dict:
    return {keys[0]: a16[:8].astype(np.float32), keys[1]: a16[8:16].astype(np.float32)}


def entity_tables(env):
    u = env.unwrapped
    robots = [a.robot for a in u.agent.agents]
    ids = sorted(u.segmentation_id_map)
    names, kinds = [], []
    for i in ids:
        o = u.segmentation_id_map[i]
        art = getattr(o, "articulation", None)
        if any(art is r for r in robots):
            kind = "gripper" if any(k in o.name.lower() for k in GRIPPER_KEYS) else "arm"
            names.append(f"robot{[art is r for r in robots].index(True)}/{o.name}")
        else:
            kind = "articulation" if art is not None else "actor"
            names.append(f"{art.name}/{o.name}" if art is not None else o.name)
        kinds.append(kind)
    return ids, names, kinds


def entity_poses(env, ids: list[int]) -> np.ndarray:
    m = env.unwrapped.segmentation_id_map
    return np.stack([to_np(m[i].pose.raw_pose)[0] for i in ids]).astype(np.float64)


def record(obs, env, ids):
    z = obs["sensor_data"]["zed2i"]
    rgb = to_np(z["rgb"])[0]
    depth = to_np(z["depth"])[0, ..., 0].astype(np.float32) / 1000.0
    seg = to_np(z["segmentation"])[0, ..., 0].astype(np.int64)
    small = cv2.resize(rgb, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return dict(jpg=np.frombuffer(buf.tobytes(), np.uint8), depth=depth[np.ix_(ROWS, COLS)],
                seg=seg[np.ix_(ROWS, COLS)], pose=entity_poses(env, ids),
                qpos=np.concatenate([to_np(a.robot.get_qpos())[0] for a in env.unwrapped.agent.agents]))


def build_track(frames, ids, kinds, K, ext):
    """part / role / delta_uvd on the 180x320 grid; landing-depth QC on moving pixels."""
    n = len(frames)
    pose = np.stack([fr["pose"] for fr in frames])                                # [T, E, 7]
    pos, rot = pose[..., :3], quat_wxyz_to_mat(pose[..., 3:])
    shift = np.linalg.norm(pos - pos[:1], axis=-1).max(0)
    cos = (np.einsum("tbij,bij->tb", rot, rot[0]) - 1.0) / 2.0
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max(0)
    ent_part = np.zeros(len(ids), np.uint8)
    for j, k in enumerate(kinds):
        if k == "arm":
            ent_part[j] = 1
        elif k == "gripper":
            ent_part[j] = 2
        elif shift[j] > 2e-3 or turn[j] > 1.0:
            ent_part[j] = 3
    lut = np.full(max(ids) + 2, -1, np.int64)
    lut[np.asarray(ids)] = np.arange(len(ids))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R, tr = ext[:, :3], ext[:, 3]                                                 # world -> cam
    uu = (COLS[None, :] + 0.5).astype(np.float64).repeat(GRID_H, 0)
    vv = (ROWS[:, None] + 0.5).astype(np.float64).repeat(GRID_W, 1)
    sx, sy = GRID_W / SRC_W, GRID_H / SRC_H

    ent = np.stack([lut[np.clip(fr["seg"], 0, len(lut) - 1)] for fr in frames])  # [T, H, W] entity index or -1
    part = np.where(ent >= 0, ent_part[np.clip(ent, 0, None)], 0).astype(np.uint8)
    role = np.select([part == 0, part <= 2], [0, 1], 2).astype(np.uint8)
    delta = np.zeros((n - STRIDE, GRID_H, GRID_W, 3), np.float16)
    land_err = []
    for t in range(n - STRIDE):
        m = role[t] > 0
        if not m.any():
            continue
        zt = frames[t]["depth"][m].astype(np.float64)
        pc = np.stack(((uu[m] - cx) / fx * zt, (vv[m] - cy) / fy * zt, zt), -1)
        pw = (pc - tr) @ R                                                         # cam -> world
        b = ent[t][m]
        local = np.einsum("nji,nj->ni", rot[t, b], pw - pos[t, b])
        pw2 = np.einsum("nij,nj->ni", rot[t + STRIDE, b], local) + pos[t + STRIDE, b]
        q = pw2 @ R.T + tr
        z2 = np.maximum(q[:, 2], 1e-3)
        u2, v2 = q[:, 0] / z2 * fx + cx, q[:, 1] / z2 * fy + cy
        d = np.stack(((u2 - uu[m]) * sx, (v2 - vv[m]) * sy, q[:, 2] - zt), -1)
        delta[t][m] = d.astype(np.float16)
        gi, gj = np.floor(v2 * sy).astype(np.int64), np.floor(u2 * sx).astype(np.int64)
        ok = (gi >= 0) & (gi < GRID_H) & (gj >= 0) & (gj < GRID_W) & (np.hypot(d[:, 0], d[:, 1]) > 1.0)
        if ok.any():
            land_err.append(np.abs(frames[t + STRIDE]["depth"][gi[ok], gj[ok]] - q[ok, 2]))
    err = np.concatenate(land_err) if land_err else np.array([np.nan])
    return part, role, delta, ent_part, (float(np.nanmedian(err)), float(np.nanmean(err < 5e-3)))


def replay(env_hi, keys, seed, actions):
    """Open-loop replay; returns per-frame records (obs before action t) and success."""
    obs, _ = env_hi.reset(seed=seed)
    ids, names, kinds = entity_tables(env_hi)
    frames, success = [record(obs, env_hi, ids)], False
    for t, a in enumerate(actions):
        obs, _, _, _, info = env_hi.step(split_action(a, keys))
        success |= bool(to_np(info["success"]).reshape(-1)[0])
        if t < len(actions) - 1:
            frames.append(record(obs, env_hi, ids))
    sp = obs["sensor_param"]["zed2i"]
    K = to_np(sp["intrinsic_cv"])[0].astype(np.float64)
    ext = to_np(sp["extrinsic_cv"])[0].astype(np.float64)
    return frames, success, ids, names, kinds, K, ext


def process_episode(env_dir, level, task, ep, data, ref, env_lo, env_hi, prev_seed, args):
    rows = data[data["episode_index"] == ep].sort_values("frame_index")
    actions = np.stack(rows["action.qpos_gripper_actions"].to_numpy()).astype(np.float32)
    qpos = np.stack(rows["observation.qpos_gripper_states"].to_numpy()).astype(np.float32)
    keys = list(env_hi.action_space.keys())
    t0 = time.time()
    t_replay = 0.0
    scanned, tried, success = {}, [], False

    def attempt(seed):
        nonlocal t_replay
        t1 = time.time()
        res = replay(env_hi, keys, seed, actions)
        t_replay += time.time() - t1
        tried.append(dict(seed=seed, mse=round(scanned[seed], 2), success=res[1]))
        return res

    # Seeds increase within a directory: scan forward and replay the first close match.
    for s in range(prev_seed + 1, prev_seed + 1 + args.seed_window):
        scanned[s] = first_frame_mse(env_lo, ref, s)
        if scanned[s] < args.mse_accept:
            frames, success, ids, names, kinds, K, ext = attempt(s)
            if success:
                break
    # Fallback: best-ranked candidates of the window, then of a wide scan from the episode index.
    for wide in (False, True):
        if success:
            break
        if wide:
            for s in range(ep, ep + args.seed_window_wide):
                if s not in scanned:
                    scanned[s] = first_frame_mse(env_lo, ref, s)
        done_seeds = {t["seed"] for t in tried}
        for mse, s in sorted((m, s) for s, m in scanned.items() if s not in done_seeds)[: args.max_seed_tries]:
            frames, success, ids, names, kinds, K, ext = attempt(s)
            if success:
                break
    t_seed = time.time() - t0 - t_replay
    ranked = sorted(scanned.values())
    rec = dict(env_dir=env_dir.name, episode=ep, frames=len(actions), seed_tries=tried, seeds_scanned=len(scanned),
               t_seed=round(t_seed, 1), t_replay=round(t_replay, 1))
    if not success:
        rec.update(ok=False, reason="replay_failed")
        return rec, prev_seed

    part, role, delta, ent_part, land = build_track(frames, ids, kinds, K, ext)
    sim_q = np.stack([fr["qpos"] for fr in frames]).astype(np.float32)
    k_grid = np.array([[K[0, 0] * GRID_W / SRC_W, 0, K[0, 2] * GRID_W / SRC_W],
                       [0, K[1, 1] * GRID_H / SRC_H, K[1, 2] * GRID_H / SRC_H], [0, 0, 1]], np.float32)
    out = Path(args.out) / env_dir.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"episode{ep:03d}.h5"
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h:
        ds = h.create_dataset("rgb", (len(frames),), dtype=h5py.vlen_dtype(np.uint8))
        for i, fr in enumerate(frames):
            ds[i] = fr["jpg"]
        h.create_dataset("depth_mm", data=np.stack([fr["depth"] for fr in frames]) * 1000.0, dtype=np.float16,
                         chunks=(1, GRID_H, GRID_W), compression="lzf")
        for name, arr in (("part", part), ("role", role)):
            h.create_dataset(name, data=arr, chunks=(1, GRID_H, GRID_W), compression="lzf")
        h.create_dataset("delta_uvd", data=delta, chunks=(1, GRID_H, GRID_W, 3), compression="lzf")
        h["qpos18"], h["actions16"], h["sim_qpos18"] = qpos, actions, sim_q
        h.create_dataset("entity_pose", data=np.stack([fr["pose"] for fr in frames]).astype(np.float32),
                         compression="lzf")
        h["intrinsic_cv"], h["extrinsic_cv"] = k_grid, ext.astype(np.float32)
        h.attrs.update(
            env_dir=env_dir.name, level=level, task=task, episode=ep, seed=tried[-1]["seed"],
            seed_mse=tried[-1]["mse"], seed_mse_next_best=round(ranked[1], 2) if len(ranked) > 1 else -1.0,
            replay_success=True, schema=SCHEMA, stride=STRIDE, frames=len(frames), height=GRID_H, width=GRID_W,
            rgb_size=f"{RGB_W}x{RGB_H}", src_size=f"{SRC_W}x{SRC_H}",
            units="du px, dv px (180x320 grid of the 1280x720 zed2i), dd m",
            part_labels="0 static, 1 arm, 2 gripper, 3 movable object",
            role_labels="0 static, 1 robot (arm+gripper), 2 movable object",
            entities=json.dumps(names), entity_kinds=json.dumps(kinds), entity_part=json.dumps(ent_part.tolist()),
            qc_landing_depth_err_median_m=land[0], qc_landing_within_5mm=land[1],
            qc_sim_vs_dataset_qpos_maxerr=float(np.abs(sim_q - qpos).max()))
    tmp.rename(path)
    rec.update(ok=True, seed=tried[-1]["seed"], land_err_median_m=land[0], land_within_5mm=land[1],
               qpos_maxerr=float(np.abs(sim_q - qpos).max()), robot_px=float((role == 1).mean()),
               object_px=float((role == 2).mean()), moving_entities=[n for n, p in zip(names, ent_part) if p == 3],
               t_total=round(time.time() - t0, 1))
    return rec, tried[-1]["seed"]


def process_dir(name: str, episodes: tuple[int, int] | None, args) -> None:
    import mani_skill.envs  # noqa: F401

    env_dir = SIM_ROOT / name
    level, env_name = name.split("_", 1)
    task = env_name.replace("-v1", "")
    set_level(level)
    env_id = env_name.replace("-v1", "L3-v1") if level == "L3" else env_name
    data, meta = load_dir(env_dir)
    all_eps = sorted(int(e) for e in meta["episode_index"])
    eps = [e for e in all_eps if episodes is None or episodes[0] <= e < episodes[1]]
    out = Path(args.out) / name
    todo = [e for e in eps if not (out / f"episode{e:03d}.h5").exists()]
    log = Path(args.out) / "_logs" / f"{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not todo:
        return
    refs = first_frames(env_dir, meta, todo)
    env_lo, env_hi = make_env(env_id, False, args.render_device), make_env(env_id, True, args.render_device)
    prev_seed = -1
    done = {}
    for e in eps:
        if (out / f"episode{e:03d}.h5").exists():
            with h5py.File(out / f"episode{e:03d}.h5", "r") as h:
                done[e] = int(h.attrs["seed"])
    for ep in eps:
        if ep in done:
            prev_seed = int(done[ep])
            continue
        if ep not in todo:
            continue
        try:
            rec, prev_seed = process_episode(env_dir, level, task, ep, data, refs[ep], env_lo, env_hi, prev_seed, args)
        except Exception as exc:  # noqa: BLE001  one broken episode must not stop the worker
            rec = dict(env_dir=name, episode=ep, ok=False, reason="exception",
                       error="".join(traceback.format_exception_only(type(exc), exc))[-400:])
            traceback.print_exc()
        rec.update(host=socket.gethostname(), pid=os.getpid(), time=time.strftime("%F %T"))
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if not rec["ok"]:
            with open(Path(args.out) / "_logs" / "bad_replay.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "seed_tries"}), flush=True)
    env_lo.close()
    env_hi.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None, help="env dirs; default all under SIM_ROOT")
    ap.add_argument("--episodes", type=int, nargs=2, default=None, help="[start, end) episode range")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--seed-window", type=int, default=40)
    ap.add_argument("--seed-window-wide", type=int, default=400)
    ap.add_argument("--mse-accept", type=float, default=25.0,
                    help="first-frame MSE below which a seed is replayed right away (matches 9-18, next best >= 33)")
    ap.add_argument("--max-seed-tries", type=int, default=3)
    ap.add_argument("--render-device", default="gpu",
                    help="pick the GPU with CUDA_VISIBLE_DEVICES; rt-fast shading hangs with an explicit cuda:N, N > 0")
    args = ap.parse_args()
    for sig in (signal.SIGTERM, signal.SIGHUP):  # run the lock-release `finally` when stopped or the tmux pane closes
        signal.signal(sig, lambda *_: sys.exit(143))

    dirs = args.dirs or sorted(p.name for p in SIM_ROOT.iterdir() if p.is_dir() and p.name[:2] in ("L0", "L1", "L2", "L3"))
    # Every worker walks all directories from its own offset; a directory is taken by mkdir lock and marked done once
    # processed, so workers balance dynamically and failed episodes are not retried by every worker.
    order = dirs[args.worker % len(dirs):] + dirs[: args.worker % len(dirs)]
    lock_root, done_root = Path(args.out) / "_locks", Path(args.out) / "_done"
    lock_root.mkdir(parents=True, exist_ok=True)
    done_root.mkdir(parents=True, exist_ok=True)
    for name in order:
        if (done_root / name).exists():
            continue
        lock = lock_root / name
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        (lock / "owner").write_text(f"{socket.gethostname()} {os.getpid()} {time.strftime('%F %T')}\n")
        try:
            if not (done_root / name).exists():
                process_dir(name, tuple(args.episodes) if args.episodes else None, args)
                if args.episodes is None:
                    (done_root / name).write_text(f"{socket.gethostname()} {os.getpid()} {time.strftime('%F %T')}\n")
        finally:
            shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
