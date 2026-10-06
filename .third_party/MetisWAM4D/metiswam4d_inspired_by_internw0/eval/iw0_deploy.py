"""XPolicyLab ``deploy`` hook for the InternW0-Delta server, run inside the official RoboDojo client (Isaac, 3.11).

Official cadence: ``get_obs`` after every executed action; one ``act`` RPC per re-plan.  The RPC acknowledges every
executed action (InternW0's step counter), attaching the post-action observation only where InternW0's memory reads
it (re-plan frames and the frames ``action_horizon`` steps before a future re-plan), so frames are not shipped for
nothing.

With ``METIS_RECORD_DIR`` the trajectory is written to ``<dir>/<variant>/layout_<id>.hdf5`` in the self-play schema of
``metiswam4d.eval.rdj_deploy`` (3-view 320x240 JPEG in RGB byte order... see there) plus, when the client enabled the
head annotators (``METIS_RECORD_4D=1``), the head camera's ground truth per frame: depth (uint16 mm), instance ids
(uint32) with the id -> prim path table, and the world poses of every rigid / articulated scene instance.
"""
import json
import os
import time

import numpy as np

from metiswam4d.eval import rdj_deploy

RECORD_DIR = os.environ.get("METIS_RECORD_DIR", "")
RECORD_4D = os.environ.get("METIS_RECORD_4D", "0") == "1"
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
TIMES: dict[str, float] = {}


class timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        TIMES[self.name] = TIMES.get(self.name, 0.0) + time.perf_counter() - self.t0


def slim(obs):
    """What the servers read: three RGB images (+ camera matrices for the robot geometry), robot state, instruction."""
    vision = {}
    for cam in CAMS:
        src = obs["vision"][cam]
        vision[cam] = {"color": np.ascontiguousarray(np.asarray(src["color"])[..., :3])}
        for key in ("intrinsic_matrix", "extrinsic_matrix"):
            if key in src:
                vision[cam][key] = np.asarray(src[key])
    return {"vision": vision, "state": obs["state"], "instruction": obs.get("instruction", "")}


def instance_poses(env):
    layout = env.scene_manager.layout_manager
    names, poses = [], []
    for name, kind in sorted(layout.instance_type_by_env[0].items()):
        if kind not in ("rigid", "articulation"):
            continue
        pos, rot = layout.get_instance_pose(env_idx=0, inst_name=name)
        if pos is None:
            continue
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        rot = rot.detach().cpu().numpy() if hasattr(rot, "detach") else np.asarray(rot)
        names.append(name)
        poses.append(np.concatenate([np.asarray(pos, np.float64).reshape(3), np.asarray(rot, np.float64).reshape(4)]))
    return names, (np.stack(poses) if poses else np.zeros((0, 7)))


def instance_labels(env):
    """``{id: prim path}`` of the head camera's instance-id annotator."""
    cm = env.camera_manager
    head = list(cm.camera_names[0]).index("cam_head")
    _, info = env.capture_manager.tiled_cameras[head].get_data("instance_id_segmentation_fast")
    return {int(k): str(v) for k, v in (info.get("idToLabels") or {}).items()}


class Recorder4D(rdj_deploy.Recorder):
    def __init__(self, env, variant, layout):
        super().__init__(env, variant, layout)
        self.gt = []
        self.labels = {}
        self.instance_names = None

    def add(self, obs):
        with timed("rec_frame"):
            super().add(obs)
        if not RECORD_4D:
            return
        with timed("rec_maps"):
            head = obs["vision"]["cam_head"]
            depth = np.asarray(head["depth"], np.float32)[::2, ::2]
            inst = np.asarray(head["instance_id"]).reshape(depth.shape[0] * 2, depth.shape[1] * 2)[::2, ::2]
        with timed("rec_poses"):
            names, poses = instance_poses(self.env)
        if self.instance_names is None:
            self.instance_names = names
        elif names != self.instance_names:
            raise RuntimeError(f"scene instances changed during the episode: {names} vs {self.instance_names}")
        with timed("rec_labels"):
            if not self.labels or len(self.gt) % 64 == 0:
                self.labels.update(instance_labels(self.env))
        self.gt.append({"depth_mm": np.clip(np.nan_to_num(depth, posinf=0.0) * 1000.0, 0, 65535).astype(np.uint16),
                        "instance": inst.astype(np.uint32), "poses": poses})

    def write(self, success, success_steps):
        super().write(success, success_steps)
        if not RECORD_4D or not self.gt:
            return
        import h5py
        path = os.path.join(RECORD_DIR, self.variant, f"layout_{self.layout:03d}.hdf5")
        with h5py.File(path, "a") as f:
            g = f.create_group("gt_head")
            g.create_dataset("depth_mm", data=np.stack([x["depth_mm"] for x in self.gt]), compression="gzip",
                             compression_opts=4, chunks=(1, *self.gt[0]["depth_mm"].shape))
            g.create_dataset("instance", data=np.stack([x["instance"] for x in self.gt]), compression="gzip",
                             compression_opts=4, chunks=(1, *self.gt[0]["instance"].shape))
            g.create_dataset("instance_poses", data=np.stack([x["poses"] for x in self.gt]))
            g.attrs["instance_names"] = json.dumps(self.instance_names or [])
            g.attrs["id_to_prim"] = json.dumps({str(k): v for k, v in sorted(self.labels.items())})
            g.attrs["note"] = ("depth = distance_to_image_plane (m*1000), instance = instance_id_segmentation_fast, "
                               "both sampled [::2, ::2] from 640x480; poses = [x y z qw qx qy qz] env-relative")
            f.attrs["schema"] = "metiswam4d_iw0.rollout4d.v1"


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    variant = os.environ.get("METIS_VARIANT", TASK_ENV.task_name)
    layout = int(TASK_ENV.env_seeds[0])
    session = f"{os.getpid()}:{variant}:{layout}:{time.time():.3f}"
    info = model_client.call(func_name="begin", obs={"session": session})
    replan, capture_mod = int(info["replan_steps"]), int(info["recent_capture_mod"])
    recorder = Recorder4D(TASK_ENV, variant, layout) if RECORD_DIR else None
    obs = recorder.head if recorder is not None else TASK_ENV.get_obs()
    if recorder is not None:
        recorder.add(obs)
    acks, steps = [slim(obs)], 0
    TIMES.clear()
    while not TASK_ENV.is_episode_end():
        with timed("act_rpc"):
            result = model_client.call(func_name="act", obs={"session": session, "acks": acks})
        actions = result["actions"]
        if not actions:
            raise RuntimeError("policy returned no actions")
        acks = []
        for i, action in enumerate(actions):
            if TASK_ENV.is_episode_end():
                break
            with timed("take_action"):
                TASK_ENV.take_action(action)
            steps += 1
            with timed("get_obs"):
                obs = TASK_ENV.get_obs()
            if recorder is not None:
                recorder.add(obs)
            needed = i + 1 == len(actions) or steps % replan == capture_mod
            acks.append(slim(obs) if needed else None)
            if steps % 25 == 0:
                print(f"[iw0_deploy] step {steps} cumulative seconds " +
                      " ".join(f"{k}={v:.1f}" for k, v in sorted(TIMES.items())), flush=True)
    model_client.call(func_name="close", obs={"session": session})
    if recorder is not None:
        recorder.write(bool(TASK_ENV.success[0]), steps)
