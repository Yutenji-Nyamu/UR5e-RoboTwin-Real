"""RT2_MetisWAM4D: clean Track4D for RoboTwin 2.0 from simulator ground truth.

Per episode this produces, under ``<OUT>/<task>/<variant>/episodeN/``:

    source.hdf5          -> symlink to the raw RoboTwin episode (RGB x3 views, sim depth, cameras, actions)
    task_instances.h5    -> symlink to the per-pixel actor-id maps rendered by the predecessor replay
    poses.npz            per-frame pose of every scene entity (actors + articulation links), joint vector
    track4d.h5           stride-4 rigid displacement field (m, head-camera frame), roles, validity
    complete.json

Sub-commands
    replay   physics-only replay (seed + planned trajectories) recording entity poses    [needs SAPIEN, GPU]
    build    depth back-projection + rigid transforms -> track4d.h5                         [CPU]
    run      replay + build for a shard of the manifest (resumable)
    manifest write the (task, variant, episode) manifest
    status   count complete / missing episodes
    norm     sample displacement statistics -> track_norm.json, depth_stats.json

Run from any directory with /usr/bin/python3.10 (SAPIEN 3.0.0b1); the RoboTwin checkout of the
predecessor project is used read-only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import zipfile

import h5py
import numpy as np

RAW = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin")
INSTANCES = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
INSTRUCTIONS = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/episode_instructions_sim_aligned.jsonl")
TEXT_CACHE = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/text_cache")
# Symlink overlay over the predecessor's RoboTwin checkout (scripts/data_prep/rt2/make_robotwin_overlay.sh).
ROBOTWIN = Path(__file__).resolve().parents[3] / "third_party" / "RoboTwin"
OUT = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D")
STRIDE = 4
VARIANTS = ("demo_clean_4d", "demo_randomized_4d")
MAX_CLUTTER_REPEAT = 40  # original workers collected ~31 episodes each (plus failed attempts)


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------


def episode_dir(task: str, variant: str, episode: int) -> Path:
    return OUT / task / variant / f"episode{episode}"


def write_manifest(variants=VARIANTS) -> Path:
    rows = []
    for task in sorted(p.name for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for variant in variants:
            data = RAW / task / variant / "data"
            if not data.exists():
                continue
            episodes = sorted(int(p.stem[len("episode"):]) for p in data.iterdir() if p.suffix == ".hdf5")
            rows.extend({"task": task, "variant": variant, "episode": e} for e in episodes)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest_episodes.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} episodes -> {path}")
    return path


def read_manifest() -> list[dict]:
    with open(OUT / "manifest_episodes.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------
# replay
# ----------------------------------------------------------------------------


def _pose_to_matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """SAPIEN pose (xyz, wxyz quaternion) -> 4x4."""
    w, x, y, z = q
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = p
    return m


class EnvCache:
    """One RoboTwin env per task, reused across episodes (planner init is paid once per task)."""

    def __init__(self, device: str):
        self.device = device
        self.envs: dict[str, object] = {}

    def get(self, task: str):
        import importlib
        os.chdir(ROBOTWIN)
        if str(ROBOTWIN) not in sys.path:
            sys.path.insert(0, str(ROBOTWIN))
        from envs._base_task import sapien
        device = self.device
        if not getattr(sapien.Engine, "_metis_device", False):
            class DeviceEngine(sapien.Engine):
                _metis_device = True

                def create_scene(self, config):
                    sapien.physx.set_scene_config(config)
                    return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device=device)])
            sapien.Engine = DeviceEngine
        if task not in self.envs:
            for other in list(self.envs):
                try:
                    self.envs.pop(other).close_env(clear_cache=False)
                except Exception:
                    pass
            self.envs[task] = getattr(importlib.import_module(f"envs.{task}"), task)()
        return self.envs[task]


class DepthMismatch(ValueError):
    """Rendered geometry differs from the recorded episode (scene not reproduced)."""


def replay_episode(task: str, variant: str, episode: int, device: str = "cuda:0", cache: EnvCache | None = None,
                   clutter_repeat: int = 1) -> dict:
    """Physics-only deterministic replay; records every entity pose at each saved frame.

    ``clutter_repeat``: RoboTwin's ``get_cluttered_table`` adds ``table_xy_bias`` to its *mutable default*
    ``xlim`` list in place, so in the original collection process the k-th episode saw the bias applied
    k times (only tasks with a non-zero bias are affected).  ``clutter_repeat = k`` reproduces that state.
    """
    import yaml
    os.chdir(ROBOTWIN)
    if str(ROBOTWIN) not in sys.path:
        sys.path.insert(0, str(ROBOTWIN))
    from script.collect_data import get_embodiment_config
    from envs import CONFIGS_PATH

    dest = episode_dir(task, variant, episode)
    dest.mkdir(parents=True, exist_ok=True)
    source = RAW / task / variant / "data" / f"episode{episode}.hdf5"
    instances = INSTANCES / task / variant / f"episode{episode}" / "task_instances.h5"
    if not source.exists():
        raise FileNotFoundError(source)
    link = dest / "source.hdf5"
    if not link.is_symlink():
        link.symlink_to(source)
    # Per-pixel actor ids: reuse the predecessor's render when it exists, otherwise render them here.
    render_ids = not instances.exists()
    if not render_ids:
        link = dest / "task_instances.h5"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(instances)
        instances_out = link
    else:
        instances_out = dest / "task_instances.h5"
        if instances_out.is_symlink():
            instances_out.unlink()

    variant_dir = RAW / task / variant
    cfg_name = "demo_randomized" if "randomized" in variant else "demo_clean"
    cfg = yaml.safe_load((ROBOTWIN / f"task_config/{cfg_name}.yml").read_text())
    embodiment = yaml.safe_load((Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text())
    robot_file = embodiment[cfg["embodiment"][0]]["file_path"]
    cfg.update(task_name=task, task_config=variant, save_path=str(variant_dir), need_plan=False, render_freq=0,
               save_data=True, eval_video_log=False, left_robot_file=robot_file, right_robot_file=robot_file,
               dual_arm_embodied=True, embodiment_name=cfg["embodiment"][0],
               left_embodiment_config=get_embodiment_config(robot_file),
               right_embodiment_config=get_embodiment_config(robot_file))
    cfg["data_type"].update(rgb=False, depth=render_ids, qpos=True, endpose=True, pointcloud=False, third_view=False,
                            mesh_segmentation=False, actor_segmentation=False)
    seed = int((variant_dir / "seed.txt").read_text().split()[episode])

    cache = cache or EnvCache(device)
    env = cache.get(task)
    for name in ("get_cluttered_table", "load_actors", "_take_picture", "setup_scene"):
        env.__dict__.pop(name, None)
    if render_ids:
        # Single-primary-sample ray-tracing shader: Segmentation / Position equal the original pixel
        # ray (prepared by the predecessor project); orders of magnitude faster than the full RT shader.
        from envs._base_task import sapien
        shader = INSTANCES / "task_mask_shader"
        if not shader.exists():
            raise FileNotFoundError(shader)
        setup_scene = env.setup_scene

        def setup_mask_scene(*args, **kwargs):
            setup_scene(*args, **kwargs)
            sapien.render.set_camera_shader_dir(str(shader))
        env.setup_scene = setup_mask_scene
    robot_ids: set[int] = set()
    task_ids: dict[int, str] = {}
    clutter_ids: set[int] = set()
    clutter = env.get_cluttered_table

    def load_clutter():
        before = {int(e.per_scene_id) for e in env.scene.entities}
        bx, by = env.table_xy_bias
        shift = clutter_repeat - 1
        result = clutter(xlim=[-.59 + bx * shift, .59 + bx * shift], ylim=[-.34 + by * shift, .34 + by * shift], zlim=[.741])
        clutter_ids.update(int(e.per_scene_id) for e in env.scene.entities if int(e.per_scene_id) not in before)
        return result
    env.get_cluttered_table = load_clutter
    load = env.load_actors

    def load_actors():
        before = {int(e.per_scene_id) for e in env.scene.entities}
        for articulation in env.scene.get_all_articulations():
            robot_ids.update(int(link.entity.per_scene_id) for link in articulation.get_links())
        load()
        task_ids.update({int(e.per_scene_id): e.name for e in env.scene.entities if int(e.per_scene_id) not in before})
    env.load_actors = load_actors

    with h5py.File(source) as src:
        n = int(src["joint_action/vector"].shape[0])
        source_joints = src["joint_action/vector"][:]
        cam_extrinsic = src["observation/head_camera/extrinsic_cv"][:]
        h, w = src["observation/head_camera/depth"].shape[1:]
        source_depth = src["observation/head_camera/depth"] if render_ids else None
        source_depth = source_depth[:] if render_ids else None
    frames: list[np.ndarray] = []
    ids_per_frame: list[np.ndarray] = []
    joint_errors: list[float] = []
    depth_errors: list[float] = []
    seg_frames = np.zeros((n, h, w), dtype=np.uint16) if render_ids else None

    def select_head_camera(cameras):
        head = cameras.static_camera_list[cameras.static_camera_name.index("head_camera")]
        cameras.collect_wrist_camera = False
        cameras.static_camera_list = [head]
        cameras.static_camera_name = ["head_camera"]
        cameras.head_camera_id = 0

    def capture(self):
        index = self.FRAME_IDX
        if index >= n:
            raise ValueError("replay produced more frames than the source episode")
        if render_ids:
            select_head_camera(self.cameras)
        obs = self.get_obs()
        err = float(np.max(np.abs(obs["joint_action"]["vector"] - source_joints[index])))
        if err > 1e-5:
            raise ValueError(f"joint mismatch at frame {index}: {err}")
        joint_errors.append(err)
        if render_ids:
            seg_frames[index] = self.cameras.get_actor_seg_ids()["head_camera"]["actor_seg_ids"]
            rendered = obs["observation"]["head_camera"]["depth"].astype(np.float32)
            gt = source_depth[index].astype(np.float32)
            p99 = float(np.percentile(np.abs(rendered - gt), 99))
            depth_errors.append(p99)
            if p99 > 2.0:  # mm; float16 source depth has ~0.25 mm half-ULP here
                raise DepthMismatch(f"rendered depth mismatch at frame {index}: P99 {p99:.2f} mm")
        ents = list(env.scene.entities)
        ids = np.array([int(e.per_scene_id) for e in ents], dtype=np.int32)
        poses = np.empty((len(ents), 7), dtype=np.float64)
        for i, e in enumerate(ents):
            pose = e.pose
            poses[i, :3] = pose.p
            poses[i, 3:] = pose.q
        ids_per_frame.append(ids)
        frames.append(poses)
        self.FRAME_IDX += 1

    env._take_picture = types.MethodType(capture, env)
    started = time.monotonic()
    try:
        env.setup_demo(now_ep_num=episode, seed=seed, **cfg)
        trajectory = env.load_tran_data(episode)
        cfg["left_joint_path"] = trajectory["left_joint_path"]
        cfg["right_joint_path"] = trajectory["right_joint_path"]
        env.set_path_lst(cfg)
        env.play_once()
        if env.FRAME_IDX != n:
            raise ValueError(f"replay frame count {env.FRAME_IDX} != source {n}")
    finally:
        # Instance-level patches must not leak into the next episode of the cached env.
        for name in ("get_cluttered_table", "load_actors", "_take_picture", "setup_scene"):
            env.__dict__.pop(name, None)
        cache.count = getattr(cache, "count", 0) + 1
        env.close_env(clear_cache=cache.count % 5 == 0)  # scene closed, Robot/planner kept (as collect_data.py)

    all_ids = sorted({int(i) for ids in ids_per_frame for i in ids})
    index_of = {i: k for k, i in enumerate(all_ids)}
    poses = np.full((n, len(all_ids), 7), np.nan, dtype=np.float32)
    for t, (ids, p) in enumerate(zip(ids_per_frame, frames)):
        for i, row in zip(ids, p):
            poses[t, index_of[int(i)]] = row
    if render_ids:
        tmp = dest / "task_instances.tmp.h5"
        with h5py.File(tmp, "w") as inst:
            inst.create_dataset("head_camera", data=seg_frames, chunks=(1, h, w), compression="gzip", compression_opts=4)
            inst.attrs.update(complete=True, seed=seed, task_entities_json=json.dumps({int(k): v for k, v in task_ids.items()}),
                              robot_ids=sorted(robot_ids), clutter_ids=sorted(clutter_ids), selected_ids=sorted(task_ids),
                              method="metiswam4d replay with the predecessor's single-sample RT shader",
                              depth_p99_mm_max=max(depth_errors) if depth_errors else -1.0)
        os.replace(tmp, instances_out)
        selected = sorted(task_ids)
        inst_ids = np.unique(seg_frames[0])
        inst_task = {str(k): v for k, v in task_ids.items()}
    else:
        with h5py.File(instances) as inst:
            inst_task = json.loads(inst.attrs.get("task_entities_json", "{}"))
            selected = [int(i) for i in np.atleast_1d(inst.attrs.get("selected_ids", []))]
            inst_ids = np.unique(inst["head_camera"][0])
    missing = sorted(set(int(i) for i in inst_ids if i != 0) - set(all_ids))
    if missing:
        raise ValueError(f"instance ids {missing} never appeared in the replayed scene")
    if set(map(int, inst_task)) != set(task_ids):
        raise ValueError(f"task entity ids differ from the instance file: {sorted(task_ids)} vs {sorted(map(int, inst_task))}")
    np.savez_compressed(
        dest / "poses.npz", ids=np.array(all_ids, dtype=np.int32), poses=poses, clutter_repeat=clutter_repeat,
        robot_ids=np.array(sorted(robot_ids), dtype=np.int32), task_ids=np.array(sorted(task_ids), dtype=np.int32),
        clutter_ids=np.array(sorted(clutter_ids), dtype=np.int32), selected_ids=np.array(selected, dtype=np.int32),
        joint_vector=source_joints.astype(np.float32), seed=seed, frames=n,
        task_names=json.dumps({int(k): v for k, v in task_ids.items()}),
        head_extrinsic_cv=cam_extrinsic.astype(np.float32),
    )
    return {"frames": n, "entities": len(all_ids), "robot_links": len(robot_ids), "task_entities": len(task_ids),
            "clutter": len(clutter_ids), "max_joint_error": max(joint_errors), "rendered_ids": render_ids,
            "clutter_repeat": clutter_repeat,
            "depth_p99_mm_max": max(depth_errors) if depth_errors else None, "replay_seconds": time.monotonic() - started}


# ----------------------------------------------------------------------------
# build track4d
# ----------------------------------------------------------------------------


def _relative_transforms(poses: np.ndarray, t: int, t_next: int) -> np.ndarray:
    """``[N, 4, 4]`` world transforms mapping entity points at frame t to frame t_next."""
    n = poses.shape[1]
    out = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    for i in range(n):
        a, b = poses[t, i], poses[t_next, i]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        out[i] = _pose_to_matrix(b[:3], b[3:]) @ np.linalg.inv(_pose_to_matrix(a[:3], a[3:]))
    return out


def build_track4d(task: str, variant: str, episode: int, motion_threshold_m: float = 0.002) -> dict:
    dest = episode_dir(task, variant, episode)
    data = np.load(dest / "poses.npz", allow_pickle=False)
    ids, poses = data["ids"], data["poses"].astype(np.float64)
    robot_ids, task_ids, clutter_ids = set(data["robot_ids"].tolist()), set(data["task_ids"].tolist()), set(data["clutter_ids"].tolist())
    n = int(data["frames"])
    with h5py.File(dest / "source.hdf5") as src, h5py.File(dest / "task_instances.h5") as inst:
        depth_mm = src["observation/head_camera/depth"]
        K = src["observation/head_camera/intrinsic_cv"][:].astype(np.float64)       # [T, 3, 3]
        E = src["observation/head_camera/extrinsic_cv"][:].astype(np.float64)       # [T, 3, 4]
        inst_ids = inst["head_camera"]
        h, w = depth_mm.shape[1:]

        # Object role: task entities plus anything that moves during the episode (excluding the robot).
        centre = poses[:, :, :3]
        moved = np.nanmax(np.linalg.norm(centre - centre[:1], axis=-1), axis=0) > motion_threshold_m
        object_ids = set(task_ids) | {int(i) for i, m in zip(ids, moved) if m and int(i) not in robot_ids}
        role_of = np.zeros(int(ids.max()) + 2, dtype=np.uint8)
        for i in robot_ids:
            role_of[i] = 1
        for i in object_ids:
            if i < len(role_of):
                role_of[i] = 2
        index_of = np.full(int(ids.max()) + 2, -1, dtype=np.int64)
        index_of[ids] = np.arange(len(ids))

        vv, uu = np.mgrid[0:h, 0:w]
        pix = np.stack((uu, vv, np.ones_like(uu)), axis=-1).astype(np.float64)  # [H, W, 3]

        tmp = dest / "track4d.tmp.h5"
        with h5py.File(tmp, "w") as out:
            d_delta = out.create_dataset("delta_xyz_cam", (n - STRIDE, h, w, 3), dtype="f2", chunks=(1, h, w, 3),
                                         compression="gzip", compression_opts=4)
            d_valid = out.create_dataset("valid", (n - STRIDE, h, w), dtype="u1", chunks=(1, h, w), compression="gzip")
            d_role = out.create_dataset("role", (n, h, w), dtype="u1", chunks=(1, h, w), compression="gzip")
            stats = {"abs_max": np.zeros(3), "abs_p999": [], "object_pixels": 0, "robot_pixels": 0}
            for t in range(n):
                seg = inst_ids[t].astype(np.int64)
                known = (seg > 0) & (seg < len(index_of)) & (index_of[np.clip(seg, 0, len(index_of) - 1)] >= 0)
                d_role[t] = np.where(known, role_of[np.clip(seg, 0, len(role_of) - 1)], 0)
                if t >= n - STRIDE:
                    continue
                z = depth_mm[t].astype(np.float64) / 1000.0
                valid = known & np.isfinite(z) & (z > 1e-4)
                K_inv = np.linalg.inv(K[t])
                X_c = (pix @ K_inv.T) * z[..., None]                                  # [H, W, 3] camera (cv)
                R, tr = E[t][:, :3], E[t][:, 3]
                X_w = (X_c - tr) @ R                                                   # R^T (X_c - t)
                rel = _relative_transforms(poses, t, t + STRIDE)                      # [N, 4, 4]
                idx = np.where(valid, index_of[np.clip(seg, 0, len(index_of) - 1)], 0)
                M = rel[idx]                                                            # [H, W, 4, 4]
                X_w2 = np.einsum("hwij,hwj->hwi", M[..., :3, :3], X_w) + M[..., :3, 3]
                X_c2 = X_w2 @ R.T + tr
                delta = np.where(valid[..., None], X_c2 - X_c, 0.0)
                d_delta[t] = delta.astype(np.float16)
                d_valid[t] = valid.astype(np.uint8)
                fg = valid & (d_role[t] > 0)
                if fg.any():
                    mag = np.abs(delta[fg])
                    stats["abs_max"] = np.maximum(stats["abs_max"], mag.max(axis=0))
                    stats["abs_p999"].append(np.percentile(mag, 99.9, axis=0))
                stats["object_pixels"] += int((valid & (d_role[t] == 2)).sum())
                stats["robot_pixels"] += int((valid & (d_role[t] == 1)).sum())
            out.attrs.update(
                stride=STRIDE, frames=n, camera="head_camera", units="m, source-frame cv camera coordinates",
                roles="0 background, 1 robot, 2 object", object_ids=sorted(object_ids), robot_ids=sorted(robot_ids),
                task_ids=sorted(task_ids), moving_clutter_ids=sorted(object_ids - set(task_ids)),
                motion_threshold_m=motion_threshold_m, complete=True)
        os.replace(tmp, dest / "track4d.h5")
    return {"frames": n, "object_ids": sorted(object_ids), "abs_max_m": stats["abs_max"].tolist(),
            "abs_p999_m": (np.median(stats["abs_p999"], axis=0).tolist() if stats["abs_p999"] else None),
            "object_pixels": stats["object_pixels"], "robot_pixels": stats["robot_pixels"]}


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------


def process(row: dict, device: str, log, cache: EnvCache | None = None) -> str:
    dest = episode_dir(row["task"], row["variant"], row["episode"])
    if (dest / "complete.json").exists():
        return "skipped"
    started = time.monotonic()
    report: dict = {**row}
    if not (dest / "poses.npz").exists():
        last: Exception | None = None
        for repeat in range(1, MAX_CLUTTER_REPEAT + 1):
            try:
                report["replay"] = replay_episode(row["task"], row["variant"], row["episode"], device, cache, repeat)
                break
            except Exception as exc:  # DepthMismatch, or scene generation failing (UnStableError...) for a wrong k
                last = exc
                # Only tasks with a non-zero table bias can differ by the repeat count.
                bias = getattr(cache.get(row["task"]), "table_xy_bias", [0, 0])
                if not any(abs(float(b)) > 0 for b in bias):
                    raise
                if not isinstance(exc, DepthMismatch) and "UnStable" not in type(exc).__name__ and repeat == 1:
                    raise  # a genuine error unrelated to the clutter state
                log(f"retry {row['task']}/episode{row['episode']} with clutter_repeat={repeat + 1} ({type(exc).__name__}: {str(exc)[:80]})")
        else:
            raise RuntimeError(f"no clutter_repeat in 1..{MAX_CLUTTER_REPEAT} reproduces the scene: {last!r}")
    try:
        report["track4d"] = build_track4d(row["task"], row["variant"], row["episode"])
    except (EOFError, zipfile.BadZipFile, OSError) as exc:
        # poses.npz truncated by a node reboot mid-write: replay again, then build.
        log(f"corrupt poses.npz for {row['task']}/episode{row['episode']} ({exc!r}); replaying again")
        (dest / "poses.npz").unlink(missing_ok=True)
        report["replay"] = replay_episode(row["task"], row["variant"], row["episode"], device, cache, 1)
        report["track4d"] = build_track4d(row["task"], row["variant"], row["episode"])
    report["seconds"] = time.monotonic() - started
    (dest / "complete.json").write_text(json.dumps(report, indent=1))
    (dest / "failed.json").unlink(missing_ok=True)
    (dest / "failed.json").unlink(missing_ok=True)
    log(f"done {row['task']}/{row['variant']}/episode{row['episode']} in {report['seconds']:.1f}s "
        f"objects={report['track4d']['object_ids']} p999={report['track4d']['abs_p999_m']}")
    return "done"


def run_shard(shard: int, num_shards: int, device: str, variants: tuple[str, ...], limit: int | None,
              reverse: bool = False) -> None:
    """``reverse``: process the shard's block back-to-front; a second node can mirror the same shard ids so
    both ends of every block are consumed (each episode is skipped once ``complete.json`` exists)."""
    # Pending episodes ordered variant-major (as given) then task / episode; each shard takes one
    # contiguous block, so it touches only 1-2 tasks (planner init is paid per task) while every
    # shard has work regardless of how many tasks remain.
    order = {v: i for i, v in enumerate(variants)}
    pending = OUT / "pending.jsonl"
    if pending.exists() and time.time() - pending.stat().st_mtime < 3600:
        # Written once by the launcher (``pending`` sub-command): avoids N processes scanning Ceph.
        with open(pending) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rows = [r for r in rows if r["variant"] in variants]
    else:
        rows = [r for r in read_manifest() if r["variant"] in variants]
        rows = [r for r in rows if not (episode_dir(r["task"], r["variant"], r["episode"]) / "complete.json").exists()]
    # Each variant is chunked separately so every shard first finishes its share of the earlier
    # variant (clean) before moving on; within a variant a shard's block is contiguous in (task, episode).
    picked: list[dict] = []
    for variant in variants:
        subset = sorted((r for r in rows if r["variant"] == variant), key=lambda r: (r["task"], r["episode"]))
        n = len(subset)
        picked.extend(subset[(shard * n) // num_shards:((shard + 1) * n) // num_shards])
    rows = picked[::-1] if reverse else picked
    mine = sorted({r["task"] for r in rows})
    if limit:
        rows = rows[:limit]
    cache = EnvCache(device)
    log_dir = OUT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard{shard:02d}_of{num_shards:02d}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"shard {shard}/{num_shards}: {len(rows)} episodes of tasks {sorted(mine)} on {device}")
    done = skipped = failed = 0
    for row in rows:
        try:
            status = process(row, device, log, cache)
            done += status == "done"
            skipped += status == "skipped"
        except Exception as exc:  # keep the shard alive; record the failure
            failed += 1
            # A broken env (e.g. planner kernel build failed) must not poison the following episodes.
            broken = cache.envs.pop(row["task"], None)
            if broken is not None:
                try:
                    broken.close_env(clear_cache=True)
                except Exception:
                    pass
            dest = episode_dir(row["task"], row["variant"], row["episode"])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "failed.json").write_text(json.dumps({**row, "error": repr(exc), "trace": traceback.format_exc()}, indent=1))
            log(f"FAILED {row['task']}/{row['variant']}/episode{row['episode']}: {exc!r}")
    log(f"shard finished: done={done} skipped={skipped} failed={failed}")


def status(variants: tuple[str, ...]) -> None:
    rows = [r for r in read_manifest() if r["variant"] in variants]
    complete = failed = 0
    per_variant: dict[str, list[int]] = {}
    for r in rows:
        d = episode_dir(r["task"], r["variant"], r["episode"])
        c, f = (d / "complete.json").exists(), (d / "failed.json").exists()
        complete += c
        failed += f and not c
        per_variant.setdefault(r["variant"], [0, 0])
        per_variant[r["variant"]][0] += c
        per_variant[r["variant"]][1] += 1
    print(json.dumps({"total": len(rows), "complete": complete, "failed": failed, "per_variant": per_variant}, indent=1))


def write_pending(variants: tuple[str, ...]) -> None:
    """pending.jsonl: episodes without complete.json (one scan, shared by all shards of a launch)."""
    rows = [r for r in read_manifest() if r["variant"] in variants
            and not (episode_dir(r["task"], r["variant"], r["episode"]) / "complete.json").exists()]
    tmp = OUT / "pending.jsonl.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, OUT / "pending.jsonl")
    print(f"{len(rows)} pending episodes -> {OUT / 'pending.jsonl'}")


def write_index(variants: tuple[str, ...]) -> None:
    """index.jsonl of complete episodes (task, variant, episode, frames) for the training dataset."""
    rows = [r for r in read_manifest() if r["variant"] in variants]
    out_rows = []
    bad: list[dict] = []
    for r in rows:
        d = episode_dir(r["task"], r["variant"], r["episode"])
        c = d / "complete.json"
        if c.exists():
            try:
                frames = json.loads(c.read_text())["track4d"]["frames"]
            except (json.JSONDecodeError, KeyError, TypeError):
                # Truncated by a node reboot mid-write: drop the marker so the episode is redone.
                c.unlink()
                bad.append(r)
                continue
            out_rows.append({**r, "frames": int(frames)})
    if bad:
        with open(OUT / "pending.jsonl", "w") as f:
            for r in bad:
                f.write(json.dumps(r) + "\n")
        print(f"{len(bad)} truncated complete.json removed -> pending.jsonl (rerun a shard, then index again)")
    tmp = OUT / "index.jsonl.tmp"
    with open(tmp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, OUT / "index.jsonl")
    print(f"{len(out_rows)} complete episodes -> {OUT / 'index.jsonl'}")


def compute_norm(sample_per_task: int, variants: tuple[str, ...]) -> None:
    """Global P99.5 per axis over foreground displacement, and head-depth statistics."""
    rows = [r for r in read_manifest() if r["variant"] in variants]
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    disp: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    used = 0
    for task, items in sorted(by_task.items()):
        for r in items[:sample_per_task]:
            d = episode_dir(task, r["variant"], r["episode"])
            if not (d / "complete.json").exists():
                continue
            with h5py.File(d / "track4d.h5") as f, h5py.File(d / "source.hdf5") as src:
                n = f["delta_xyz_cam"].shape[0]
                for t in range(0, n, 5):
                    fg = (f["valid"][t] > 0) & (f["role"][t] > 0)
                    if fg.any():
                        disp.append(np.abs(f["delta_xyz_cam"][t][fg].astype(np.float32)))
                    z = src["observation/head_camera/depth"][t].astype(np.float32) / 1000.0
                    depths.append(z[z > 1e-4][::7])
            used += 1
    disp_all = np.concatenate(disp) if disp else np.zeros((1, 3), np.float32)
    depth_all = np.concatenate(depths) if depths else np.zeros(1, np.float32)
    norm = {"scale_m": np.percentile(disp_all, 99.5, axis=0).tolist(), "mu": 31, "percentile": 99.5,
            "episodes": used, "foreground_pixels": int(disp_all.shape[0]), "stride": STRIDE,
            "abs_p50": np.percentile(disp_all, 50, axis=0).tolist(), "abs_p99": np.percentile(disp_all, 99, axis=0).tolist()}
    depth_stats = {"head_camera": {"min_m": float(np.percentile(depth_all, 0.1)), "max_m": float(np.percentile(depth_all, 99.9)),
                                   "p50_m": float(np.percentile(depth_all, 50))}, "episodes": used}
    (OUT / "track_norm.json").write_text(json.dumps(norm, indent=1))
    (OUT / "depth_stats.json").write_text(json.dumps(depth_stats, indent=1))
    print(json.dumps({"track_norm": norm, "depth_stats": depth_stats}, indent=1))


def link_shared() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, target in (("episode_instructions_sim_aligned.jsonl", INSTRUCTIONS), ("text_cache", TEXT_CACHE)):
        link = OUT / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    print("linked instructions and text cache")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    sub.add_parser("link")
    p = sub.add_parser("run")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reverse", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p = sub.add_parser("norm")
    p.add_argument("--sample-per-task", type=int, default=2)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p = sub.add_parser("index")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p = sub.add_parser("pending")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p = sub.add_parser("one")
    p.add_argument("task")
    p.add_argument("variant")
    p.add_argument("episode", type=int, nargs="+")
    p.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.cmd == "manifest":
        write_manifest()
    elif args.cmd == "link":
        link_shared()
    elif args.cmd == "run":
        run_shard(args.shard, args.num_shards, args.device, tuple(args.variants), args.limit, args.reverse)
    elif args.cmd == "status":
        status(tuple(args.variants))
    elif args.cmd == "norm":
        compute_norm(args.sample_per_task, tuple(args.variants))
    elif args.cmd == "index":
        write_index(tuple(args.variants))
    elif args.cmd == "pending":
        write_pending(tuple(args.variants))
    elif args.cmd == "one":
        cache = EnvCache(args.device)
        for episode in args.episode:
            t0 = time.monotonic()
            result = process({"task": args.task, "variant": args.variant, "episode": episode}, args.device, print, cache)
            print(f"episode {episode}: {result} in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
