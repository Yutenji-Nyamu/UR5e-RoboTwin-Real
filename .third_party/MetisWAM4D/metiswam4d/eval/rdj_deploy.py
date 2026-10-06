"""XPolicyLab ``deploy`` hook run inside the official RoboDojo evaluation client (Isaac Sim, Python 3.11).

One ``act`` call per re-plan: the observation (plus ``variant`` / ``layout_id`` / ``replan`` and the per-variant
``policy`` options) goes to the policy server, the returned ``take_action`` dictionaries are executed one by one at the
official 25 Hz cadence; the episode stops as soon as the environment reports its end.

Observation cadence: by default one observation per executed chunk (``render_sync`` kit flags make it current).  With
``METIS_OBS_EVERY_STEP=1`` an observation is taken after every action (the official deploy cadence; ``get_obs`` also
steps the Kit app, so this changes the physics settling between actions).  With ``METIS_RECORD_DIR`` set the per-step
cadence is implied and the trajectory is
written to ``<dir>/<variant>/layout_<id>.hdf5``: 3-view JPEG (320x240, ``cv2.imencode`` on the RGB array, the corpus
byte order), joint states, link6 poses (xyz + wxyz), grippers, per-frame process score, outcome — the raw material of
the self-play dataset (``scripts/data_prep/robodojo/rdj_rollout_dataset.py``).
"""
import json
import os
import time

import numpy as np

POLICY = json.loads(os.environ.get("METIS_POLICY", "{}") or "{}")
RECORD_DIR = os.environ.get("METIS_RECORD_DIR", "")
OBS_EVERY_STEP = bool(RECORD_DIR) or os.environ.get("METIS_OBS_EVERY_STEP", "0") == "1"


def _frame(obs, score):
    import cv2
    jpegs = []
    for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        rgb = np.asarray(obs["vision"][cam]["color"])[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        jpegs.append(buf.tobytes())
    s = obs["state"]
    return {
        "jpeg": jpegs,
        "qpos": np.concatenate([np.asarray(s["left_arm_joint_state"], np.float32).reshape(6),
                                np.asarray(s["left_ee_joint_state"], np.float32).reshape(-1)[:1],
                                np.asarray(s["right_arm_joint_state"], np.float32).reshape(6),
                                np.asarray(s["right_ee_joint_state"], np.float32).reshape(-1)[:1]]),
        "ee": np.concatenate([np.asarray(s["left_ee_pose"], np.float32).reshape(7),
                              np.asarray(s["right_ee_pose"], np.float32).reshape(7)]),
        "score": float(score),
    }


def _score(env):
    try:
        values = env.reward_manager.get_score()
        return float(values[0]) if values is not None else float("nan")
    except Exception:  # noqa: BLE001 - the score is a diagnostic, never fails the episode
        return float("nan")


class Recorder:
    def __init__(self, env, variant, layout):
        import h5py
        self.h5py = h5py
        self.env, self.variant, self.layout = env, variant, layout
        self.frames = []
        self.head = env.get_obs()
        self.camera = {cam: (np.asarray(self.head["vision"][cam]["intrinsic_matrix"], np.float64),
                             np.asarray(self.head["vision"][cam]["extrinsic_matrix"], np.float64),
                             tuple(np.asarray(self.head["vision"][cam]["color"]).shape[:2]))
                       for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
        self.instruction = str(self.head.get("instruction", ""))
        self.started = time.time()

    def add(self, obs):
        self.frames.append(_frame(obs, _score(self.env)))

    def write(self, success, success_steps):
        if not self.frames:
            return
        path = os.path.join(RECORD_DIR, self.variant, f"layout_{self.layout:03d}.hdf5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        h5py = self.h5py
        with h5py.File(tmp, "w") as f:
            f.attrs.update(variant=self.variant, task=str(self.env.task_name), layout_id=int(self.layout),
                           instruction=self.instruction, success=bool(success), frames=len(self.frames),
                           success_steps=int(success_steps), final_score=float(self.frames[-1]["score"]),
                           frequency=25, seconds=round(time.time() - self.started, 1), policy=json.dumps(POLICY),
                           schema="metiswam4d.rdj_rollout.v1")
            vlen = h5py.vlen_dtype(np.dtype("uint8"))
            for i, cam in enumerate(("head_camera", "left_camera", "right_camera")):
                g = f.create_group(f"observation/{cam}")
                d = g.create_dataset("rgb", (len(self.frames),), dtype=vlen)
                for t, fr in enumerate(self.frames):
                    d[t] = np.frombuffer(fr["jpeg"][i], dtype=np.uint8)
                live = ("cam_head", "cam_left_wrist", "cam_right_wrist")[i]
                k, c2w, shape = self.camera[live]
                g.create_dataset("intrinsic_live", data=k)
                g.create_dataset("cam2world_gl", data=c2w)
                g.create_dataset("live_shape", data=np.asarray(shape))
            f.create_dataset("joint_state/vector", data=np.stack([fr["qpos"] for fr in self.frames]))
            f.create_dataset("state/ee_poses_wxyz", data=np.stack([fr["ee"] for fr in self.frames]))
            f.create_dataset("score", data=np.asarray([fr["score"] for fr in self.frames], np.float32))
        os.replace(tmp, path)


def execute_chunk(env, actions, recorder=None):
    executed = 0
    last_obs = None
    for action in actions:
        if env.is_episode_end():
            break
        env.take_action(action)
        executed += 1
        if OBS_EVERY_STEP:
            last_obs = env.get_obs()
            if recorder is not None:
                recorder.add(last_obs)
    return executed, last_obs


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    variant = os.environ.get("METIS_VARIANT", TASK_ENV.task_name)
    layout = int(TASK_ENV.env_seeds[0])
    recorder = Recorder(TASK_ENV, variant, layout) if RECORD_DIR else None
    if recorder is not None:
        recorder.add(recorder.head)
    replan, prev, executed, steps = 0, None, 0, 0
    obs = recorder.head if recorder is not None else None
    while not TASK_ENV.is_episode_end():
        if obs is None:
            obs = TASK_ENV.get_obs()
        obs["task"] = TASK_ENV.task_name
        obs["variant"] = variant
        obs["layout_id"] = layout
        obs["replan"] = replan
        obs["policy"] = POLICY
        if prev is not None:
            obs["prev_eef20"] = prev
            obs["executed"] = executed
        result = model_client.call(func_name="act", obs=obs)
        actions = result["actions"] if isinstance(result, dict) else result
        if not actions:
            raise RuntimeError("policy returned no actions")
        prev = np.asarray(result["eef20"], np.float32) if isinstance(result, dict) and "eef20" in result else None
        executed, obs = execute_chunk(TASK_ENV, actions, recorder)
        steps += executed
        replan += 1
    if recorder is not None:
        recorder.write(bool(TASK_ENV.success[0]), steps)
