"""VLABench primitive demos re-generated with simulator-exact geometry for MetisWAM4D.

Same scene sampler, expert skill sequences and success filter as the official
``scripts/trajectory_generation.py`` (the source of ``vlabench_primitive_ft_lerobot_video``); only the observation
recorder is replaced.  Per frame it keeps the three policy views (``forward`` -> head, ``franka/Franka_wrist_cam``
-> left wrist, ``right`` -> right wrist, the OpenWAM VLABench camera map) and, for the head camera, z-depth, geom
segmentation and the pose of every body.  Every head pixel belongs to one rigid body, so its 3-D motion is exact:

    X_{t+4} = T_b(t+4) T_b(t)^-1 X_t ,     X_t back-projected from depth, b = geom_bodyid[seg(u, v)]

The track grid is the head image stretched 480x480 -> 240x320, the same anisotropic resize that puts the head view
into the 256x320 canvas slot, so (du, dv) are in pixels of that stretched image.

Output ``<out>/<task>/episode<k>.h5`` (written to a temp name, renamed when complete):
    head_rgb / left_rgb / right_rgb  [T] JPEG bytes (RGB), 480x480
    head_depth_mm                    [T, 240, 320] float16
    part                             [T, 240, 320] uint8  0 static / 1 arm / 2 gripper / 3 movable object
    role                             [T, 240, 320] uint8  0 static / 1 robot (arm + gripper) / 2 object
    delta_uvd                        [T-4, 240, 320, 3] float16  (du px, dv px, dd m), 0 where role == 0
    ee_state [T, 8] (world xyz, wxyz quat, open flag)  q_state [T, 9]  trajectory [T, 8] (robot-frame waypoints)
    state7 / actions7 [T, 7]         exactly what ``scripts/convert_to_lerobot.py`` writes to ``state`` / ``actions``
    intrinsic_cv [3, 3] (240x320 grid)  extrinsic_cv [3, 4] (world -> camera, OpenCV)
    attrs: task, instruction, episode_config, entities, target_entity, schema, stride, qc_*
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
import zlib
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from VLABench.envs import load_env
from VLABench.robots import *  # noqa: F401,F403  (robot registration)
from VLABench.tasks import *  # noqa: F401,F403  (task registration)

SCHEMA = "metiswam4d.vlabench_track4d.uvd.objects.v2"
SRC = 480
GRID_H, GRID_W = 240, 320
STRIDE = 4
CAMERAS = {"head": "forward", "left": "franka/Franka_wrist_cam", "right": "right"}
MJ_OBJ_GEOM = 5
GRIPPER_KEYS = ("hand", "finger")

ROWS = np.floor((np.arange(GRID_H) + 0.5) * SRC / GRID_H).astype(np.int64)
COLS = np.floor((np.arange(GRID_W) + 0.5) * SRC / GRID_W).astype(np.int64)


class Frame(dict):
    """Recorded frame; keys it does not carry (the grasp planner reads ``masked_point_cloud``) come from the
    original full observation of the same physics state."""

    def __init__(self, data: dict, full_observation):
        super().__init__(data)
        self._full = full_observation

    def __missing__(self, key):
        return self._full()[key]


class Recorder:
    """Drop-in for ``env.get_observation``: the skill library only collects what it returns."""

    def __init__(self, env):
        self.env = env
        self.full = env.get_observation
        model = env.physics.model
        self.cam = {k: model.camera(name).id for k, name in CAMERAS.items()}

    def __call__(self, require_pcd: bool = True) -> dict:
        phys = self.env.physics
        rgb = {k: phys.render(camera_id=c, height=SRC, width=SRC) for k, c in self.cam.items()}
        depth = phys.render(camera_id=self.cam["head"], height=SRC, width=SRC, depth=True)
        seg = phys.render(camera_id=self.cam["head"], height=SRC, width=SRC, segmentation=True)
        return Frame(dict(
            rgb=rgb,
            depth=depth[np.ix_(ROWS, COLS)].astype(np.float32),
            seg=seg[np.ix_(ROWS, COLS)].astype(np.int32),
            xpos=phys.data.xpos.copy(), xmat=phys.data.xmat.copy().reshape(-1, 3, 3),
            cam_pos=phys.data.cam_xpos[self.cam["head"]].copy(),
            cam_mat=phys.data.cam_xmat[self.cam["head"]].copy().reshape(3, 3),
            ee_state=np.asarray(self.env.robot.get_ee_state(phys), np.float32),
            q_state=np.asarray(self.env.robot.get_qpos(phys), np.float32),
        ), self.full)


def body_parts(env, frames: list[dict]) -> np.ndarray:
    """Per-body part label: 0 static, 1 arm, 2 gripper, 3 non-robot body that moves during the episode (> 2 mm
    or > 1 deg).  Free bodies that never move (the table, an untouched plate) are static geometry."""
    phys = env.physics
    model = phys.model
    xpos = np.stack([fr["xpos"] for fr in frames])
    xmat = np.stack([fr["xmat"] for fr in frames])
    shift = np.linalg.norm(xpos - xpos[:1], axis=-1).max(0)
    cos = (np.einsum("tbij,bij->tb", xmat, xmat[0]) - 1.0) / 2.0
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max(0)
    part = np.zeros(model.nbody, np.uint8)
    part[(shift > 2e-3) | (turn > 1.0)] = 3
    for body in env.robot.mjcf_model.find_all("body"):
        bid = phys.bind(body).element_id
        name = model.body(bid).name.lower()
        part[bid] = 2 if any(k in name for k in GRIPPER_KEYS) else 1
    return part


def intrinsics(env) -> tuple[float, np.ndarray]:
    fovy = np.deg2rad(env.physics.model.cam_fovy[env.physics.model.camera(CAMERAS["head"]).id])
    f = SRC / (2.0 * np.tan(fovy / 2.0))
    k_grid = np.array([[f * GRID_W / SRC, 0, GRID_W / 2], [0, f * GRID_H / SRC, GRID_H / 2], [0, 0, 1]], np.float32)
    return f, k_grid


def to_cv(points_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    """World -> OpenCV camera frame (MuJoCo cameras look down -z with +y up)."""
    p = (points_world - cam_pos) @ cam_mat
    return p * np.array([1.0, -1.0, -1.0])


def build_track(frames: list[dict], geom_body: np.ndarray, part_of_body: np.ndarray,
                f: float) -> tuple[np.ndarray, ...]:
    n = len(frames)
    c = SRC / 2.0
    uu = (COLS[None, :] + 0.5).astype(np.float64).repeat(GRID_H, 0)       # source-pixel centres of the grid
    vv = (ROWS[:, None] + 0.5).astype(np.float64).repeat(GRID_W, 1)
    part = np.zeros((n, GRID_H, GRID_W), np.uint8)
    body = np.zeros((n, GRID_H, GRID_W), np.int64)
    for t, fr in enumerate(frames):
        seg = fr["seg"]
        is_geom = (seg[..., 1] == MJ_OBJ_GEOM) & (seg[..., 0] >= 0)
        b = np.where(is_geom, geom_body[np.clip(seg[..., 0], 0, None)], 0)
        body[t] = b
        part[t] = np.where(is_geom, part_of_body[b], 0)
    role = np.select([part == 0, part <= 2], [0, 1], 2).astype(np.uint8)

    delta = np.zeros((n - STRIDE, GRID_H, GRID_W, 3), np.float16)
    land_err = []
    sx, sy = GRID_W / SRC, GRID_H / SRC
    for t in range(n - STRIDE):
        a, z = frames[t], frames[t]["depth"]
        m = role[t] > 0
        if not m.any():
            continue
        zt = z[m].astype(np.float64)
        p_cv = np.stack(((uu[m] - c) / f * zt, (vv[m] - c) / f * zt, zt), -1)
        p_mj = p_cv * np.array([1.0, -1.0, -1.0])
        pw = a["cam_pos"] + p_mj @ a["cam_mat"].T
        bt = body[t][m]
        e = frames[t + STRIDE]
        local = np.einsum("nji,nj->ni", a["xmat"][bt], pw - a["xpos"][bt])
        pw2 = np.einsum("nij,nj->ni", e["xmat"][bt], local) + e["xpos"][bt]
        q = to_cv(pw2, a["cam_pos"], a["cam_mat"])
        z2 = np.maximum(q[:, 2], 1e-3)
        u2, v2 = q[:, 0] / z2 * f + c, q[:, 1] / z2 * f + c
        d = np.stack(((u2 - uu[m]) * sx, (v2 - vv[m]) * sy, q[:, 2] - zt), -1)
        delta[t][m] = d.astype(np.float16)
        # landing check on moving pixels: where the point lands inside the t+4 grid, its depth should match the
        # rendered depth unless it became occluded
        gi = np.floor(v2 * sy).astype(np.int64)
        gj = np.floor(u2 * sx).astype(np.int64)
        ok = (gi >= 0) & (gi < GRID_H) & (gj >= 0) & (gj < GRID_W) & (np.hypot(d[:, 0], d[:, 1]) > 1.0)
        if ok.any():
            land_err.append(np.abs(e["depth"][gi[ok], gj[ok]] - q[ok, 2]))
    if not land_err:
        return part, role, delta, (float("nan"), float("nan"))
    err = np.concatenate(land_err)
    return part, role, delta, (float(np.median(err)), float((err < 5e-3).mean()))


def quat2euler(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def lerobot_columns(ee_state: np.ndarray, trajectory: np.ndarray, robot_pos: np.ndarray):
    """``scripts/convert_to_lerobot.py``: state = (base-frame xyz, xyz Euler, open flag); action = waypoint[:6] +
    (finger width > 0.03)."""
    pos = ee_state[:, :3] - robot_pos
    euler = np.stack([quat2euler(q) for q in ee_state[:, 3:7]])
    state7 = np.concatenate([pos, euler, ee_state[:, 7:8]], 1).astype(np.float32)
    actions7 = np.concatenate([trajectory[:, :6], (trajectory[:, -1:] > 0.03).astype(np.float32)], 1)
    return state7, actions7.astype(np.float32)


def jpeg(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return np.frombuffer(buf.tobytes(), np.uint8)


def use_official_layout() -> None:
    """Scene heights of the March-2025 VLABench release (the official fine-tune data and the frozen evaluation
    tracks): the current code spawns the tube stand and the book shelf at z = 0.80 and the tubes 0.05 above their stand
    holes, the evaluation track configs have the stand / shelf at 0.76 and the tubes seated at the stand (0.759)."""
    from VLABench.tasks.hierarchical_tasks.primitive.select_book_series import SelectBookConfigManager
    from VLABench.tasks.hierarchical_tasks.primitive.select_chemistry_tube_series import (
        SelectChemistryTubeConfigManager)
    from VLABench.configs.constant import name2class_xml

    def lower_container(cls):
        original = cls.load_init_containers

        def load_init_containers(self, init_container):
            original(self, init_container)
            if init_container is not None:
                self.config["task"]["components"][-1]["position"][2] = 0.76
        cls.load_init_containers = load_init_containers

    lower_container(SelectChemistryTubeConfigManager)
    lower_container(SelectBookConfigManager)
    original_objects = SelectChemistryTubeConfigManager.load_objects

    def load_objects(self, target_entity):
        original_objects(self, target_entity)
        for sub in self.config["task"]["components"][-1]["subentities"]:
            if sub["class"] == name2class_xml["tube"][0]:
                sub["position"][2] = 0.0
    SelectChemistryTubeConfigManager.load_objects = load_objects


def generate(task: str, path: Path, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    env = load_env(task, robot="franka")
    env.reset()
    episode_config = env.save()
    env.get_observation = Recorder(env)
    instruction = env.task.get_instruction()
    meta = dict(entities=list(env.task.entities.keys()), target_entity=env.task.config_manager.target_entity)

    frames, waypoints, success = [], [], False
    for skill in env.get_expert_skill_sequence():
        obs, waypoint, stage_success, task_success = skill(env)
        frames.extend(obs)
        waypoints.extend(waypoint)
        if task_success:
            success = True
            break
    t_sim = time.time() - t0
    if not success or len(frames) < 33 + STRIDE:
        env.close()
        return dict(success=False, frames=len(frames), t_sim=t_sim)

    robot_pos = np.asarray(env.robot.robot_config["position"], np.float32)
    trajectory = np.array([np.asarray(w) - np.concatenate([robot_pos, np.zeros(5)]) for w in waypoints], np.float32)
    f, k_grid = intrinsics(env)
    part, role, delta, land_err = build_track(frames, env.physics.model.geom_bodyid.copy(), body_parts(env, frames), f)
    ee_state = np.stack([fr["ee_state"] for fr in frames])
    state7, actions7 = lerobot_columns(ee_state, trajectory, robot_pos)
    r_wc = frames[0]["cam_mat"].T * np.array([1.0, -1.0, -1.0])[:, None]
    extrinsic = np.concatenate([r_wc, -r_wc @ frames[0]["cam_pos"][:, None]], 1).astype(np.float32)

    tmp = path.with_suffix(".h5.tmp")
    vlen = h5py.vlen_dtype(np.uint8)
    with h5py.File(tmp, "w") as h:
        for k in CAMERAS:
            ds = h.create_dataset(f"{k}_rgb", (len(frames),), dtype=vlen)
            for i, fr in enumerate(frames):
                ds[i] = jpeg(fr["rgb"][k])
        h.create_dataset("head_depth_mm", data=np.stack([fr["depth"] for fr in frames]) * 1000.0, dtype=np.float16,
                         chunks=(1, GRID_H, GRID_W), compression="lzf")
        for name, arr in (("part", part), ("role", role)):
            h.create_dataset(name, data=arr, chunks=(1, GRID_H, GRID_W), compression="lzf")
        h.create_dataset("delta_uvd", data=delta, chunks=(1, GRID_H, GRID_W, 3), compression="lzf")
        h["ee_state"] = ee_state
        h["q_state"] = np.stack([fr["q_state"] for fr in frames])
        h["trajectory"] = trajectory
        h["state7"] = state7
        h["actions7"] = actions7
        h["intrinsic_cv"] = k_grid
        h["extrinsic_cv"] = extrinsic
        h.attrs.update(task=task, instruction=str(instruction), episode_config=json.dumps(episode_config),
                       entities=json.dumps(meta["entities"]), target_entity=str(meta["target_entity"]),
                       robot_position=robot_pos, schema=SCHEMA, stride=STRIDE, frames=len(frames), seed=seed,
                       height=GRID_H, width=GRID_W, units="du px, dv px, dd m (240x320 stretched head grid)",
                       role_labels="0 static, 1 robot (arm+gripper), 2 movable object",
                       part_labels="0 static, 1 arm, 2 gripper, 3 movable object",
                       qc_landing_depth_err_median_m=land_err[0], qc_landing_within_5mm=land_err[1])
    tmp.rename(path)
    env.close()
    return dict(success=True, frames=len(frames), t_sim=t_sim, t_total=time.time() - t0,
                land_err_median_m=land_err[0], land_within_5mm=land_err[1],
                robot_px=float((role == 1).mean()), object_px=float((role == 2).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="/ytech_milm_intern/danglingwei/datas/VLABench/gen4d")
    ap.add_argument("--start", type=int, default=0, help="first episode index of this worker")
    ap.add_argument("--count", type=int, default=1, help="successful episodes to produce")
    ap.add_argument("--max-attempts", type=int, default=0, help="0 -> 4 x count")
    ap.add_argument("--layout", choices=["official", "current"], default="official",
                    help="official: March-2025 scene heights of the evaluation tracks (see use_official_layout)")
    args = ap.parse_args()
    if args.layout == "official":
        use_official_layout()

    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    log = Path(args.out) / "_logs" / f"{args.task}_{args.start:05d}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = args.max_attempts or 4 * args.count
    k, attempt = args.start, 0
    while k < args.start + args.count and attempt < max_attempts:
        path = out / f"episode{k}.h5"
        if path.exists():
            k += 1
            continue
        seed = ((zlib.crc32(args.task.encode()) & 0xFFF) << 20) + k * 64 + attempt
        try:
            rec = generate(args.task, path, seed)
        except Exception as exc:  # noqa: BLE001  one broken scene must not stop the worker
            rec = dict(success=False, error="".join(traceback.format_exception_only(type(exc), exc))[-400:])
            traceback.print_exc()
        attempt += 1
        rec.update(task=args.task, episode=k, attempt=attempt, seed=seed, time=time.strftime("%F %T"))
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if rec["success"]:
            k += 1


if __name__ == "__main__":
    sys.exit(main())
