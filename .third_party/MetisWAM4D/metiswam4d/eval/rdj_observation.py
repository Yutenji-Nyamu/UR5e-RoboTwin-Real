"""Live RoboDojo (Isaac Sim, ARX X5) observations in the RDJ_MetisWAM4D training domain.

Every input of the policy is produced the way the training data was built (``scripts/data_prep/robodojo/rdj_track4d.py``
on top of ``JanusTrack4d_260824/preprocess/robodojo/official_geometry.py``):

    RGB x3       640x480 render -> 320x240 ``INTER_AREA`` -> JPEG q95 written with ``cv2.imencode`` on the RGB array
                 (channel order of the stored corpus) -> ``decode_bgr_jpeg`` (PIL decode + channel reversal), i.e. the
                 same bytes -> pixels path as ``RDJEpisodeDataset``.
    depth, mask  robot-only geometry rendered with SAPIEN from the URDF, the live joint states and the live head
                 camera (``DualX5Renderer``): depth in mm on robot pixels (0 elsewhere, float16 rounding), mask =
                 rendered robot links.  Identical to the training frames (IoU 1.0, 0.0 mm on a training episode).
    EEF20        ``state/<side>_ee_pose`` = link6 pose, env-relative world frame, xyz + **wxyz** quaternion; gripper
                 ``state/<side>_ee_joint_state`` in [0, 1].  rot6d = first two rotation-matrix columns.
    prompt       ``format_prompt(instruction)`` (the RoboTwin template used for the RDJ text cache).

Actions go back the same way: EEF20 (world) -> ``left_ee_pose`` / ``right_ee_pose`` (xyz + wxyz) +
``*_ee_joint_state`` dictionaries for ``env.take_action`` (official EE control, cuRobo IK).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from metiswam4d.data.robodojo.episode_dataset import decode_bgr_jpeg
from metiswam4d.data.rt2.text_cache import format_prompt

LIVE_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERAS = ("head_camera", "left_camera", "right_camera")
TRAIN_SIZE = (320, 240)  # (width, height) of the training frames
URDF = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo/Assets/Robots/x5/X5A.urdf")
SAPIEN_ICD = Path("/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json")


# ----------------------------------------------------------------------------------------------------------------
# RGB
# ----------------------------------------------------------------------------------------------------------------

def training_rgb(color: np.ndarray) -> np.ndarray:
    """Live uint8 RGB (any size) -> the 240x320 uint8 RGB frame the dataset would hold for this render."""
    rgb = np.asarray(color)
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.shape[:2] != (TRAIN_SIZE[1], TRAIN_SIZE[0]):
        rgb = cv2.resize(rgb, TRAIN_SIZE, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", np.ascontiguousarray(rgb), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return np.asarray(decode_bgr_jpeg(jpeg.tobytes()), dtype=np.uint8)


# ----------------------------------------------------------------------------------------------------------------
# robot geometry (depth + mask) from the joint states, as in the training data
# ----------------------------------------------------------------------------------------------------------------

class RobotGeometry:
    """SAPIEN robot-only head-camera depth (mm) and mask from live joint states.  One renderer per camera
    intrinsic (constant within a task; rebuilt if it changes)."""

    def __init__(self, urdf: Path = URDF):
        import os
        os.environ.setdefault("VK_ICD_FILENAMES", str(SAPIEN_ICD))
        from preprocess.robodojo.official_geometry import DualX5Renderer  # JanusTrack4d_260824 (on PYTHONPATH)
        self._cls = DualX5Renderer
        self.urdf = Path(urdf)
        self.renderer = None
        self.intrinsic = None

    def __call__(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        head = obs["vision"]["cam_head"]
        intrinsic = np.asarray(head["intrinsic_matrix"], dtype=np.float64)
        shape = tuple(np.asarray(head["color"]).shape[:2]) + (3,)
        if self.renderer is None or not np.allclose(intrinsic, self.intrinsic) or shape != self.shape:
            self.renderer = self._cls(self.urdf, {"head_camera": (intrinsic, shape)})
            self.intrinsic, self.shape = intrinsic.copy(), shape
        q = joint_vector(obs)
        self.renderer.set_state({"left": q[:6], "right": q[7:13]}, {"left": float(q[6]), "right": float(q[13])})
        rendered, _ = self.renderer.render({"head_camera": np.asarray(head["extrinsic_matrix"], dtype=np.float32)})
        links, depth_m = rendered["head_camera"]
        depth_mm = (np.asarray(depth_m, np.float32) * 1000.0).astype(np.float16).astype(np.float32)
        return depth_mm, links > 0


def joint_vector(obs: dict) -> np.ndarray:
    """``[14]`` = left arm 6, left gripper (0..1), right arm 6, right gripper — the dataset's ``qpos`` layout."""
    state = obs["state"]
    parts = []
    for side in ("left", "right"):
        arm = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float32).reshape(-1)
        grip = np.clip(np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)[:1], 0.0, 1.0)
        if arm.shape[0] != 6:
            raise ValueError(f"{side} arm joint state must have 6 values, got {arm.shape}")
        parts += [arm, grip]
    q = np.concatenate(parts)
    if not np.isfinite(q).all():
        raise ValueError("non-finite joint state")
    return q


# ----------------------------------------------------------------------------------------------------------------
# end effector
# ----------------------------------------------------------------------------------------------------------------

def pose_wxyz_to_arm10(pose7: np.ndarray, gripper: float) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64).reshape(7)
    mat = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()          # scipy takes xyzw
    return np.concatenate((pose[:3], mat[:, 0], mat[:, 1], [float(np.clip(gripper, 0.0, 1.0))]))


def eef20_from_observation(obs: dict) -> np.ndarray:
    """World-frame EEF20 exactly as ``track4d.h5/eef20`` (link6 pose + gripper opening)."""
    state = obs["state"]
    arms = []
    for side in ("left", "right"):
        grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)[0]
        arms.append(pose_wxyz_to_arm10(state[f"{side}_ee_pose"], float(grip)))
    eef = np.concatenate(arms).astype(np.float32)
    if not np.isfinite(eef).all():
        raise ValueError("non-finite end-effector state")
    return eef


def rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt inverse of the rot6d encoding; quaternion in the simulator's wxyz order."""
    r6 = np.asarray(r6, dtype=np.float64)
    a, b = r6[..., :3], r6[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    xyzw = Rotation.from_matrix(np.stack((x, y, z), axis=-1)).as_quat()
    return xyzw[..., [3, 0, 1, 2]]


def eef20_to_action_dicts(eef20_world: np.ndarray) -> list[dict]:
    """``[N, 20]`` world EEF20 -> N ``take_action`` dictionaries (official EE control)."""
    e = np.asarray(eef20_world, dtype=np.float64)
    out = []
    for row in e:
        action = {}
        for side, start in (("left", 0), ("right", 10)):
            arm = row[start:start + 10]
            action[f"{side}_ee_pose"] = np.concatenate((arm[:3], rot6d_to_quat_wxyz(arm[3:9]))).astype(np.float32)
            action[f"{side}_ee_joint_state"] = np.clip(arm[9:10], 0.0, 1.0).astype(np.float32)
        out.append(action)
    return out


# ----------------------------------------------------------------------------------------------------------------
# the policy request
# ----------------------------------------------------------------------------------------------------------------

def instruction_of(obs: dict) -> str:
    text = obs.get("instruction")
    if isinstance(text, (list, tuple)):
        text = text[0]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the simulator instruction must be a non-empty string")
    return text.strip()


def request_from_observation(obs: dict, geometry: RobotGeometry, seed: int, policy: dict | None = None,
                             full: bool = False) -> dict:
    """The ``RDJPolicy.predict_batch`` request for one live observation."""
    vision = obs["vision"]
    rgb = {train: training_rgb(vision[live]["color"]) for live, train in zip(LIVE_CAMERAS, CAMERAS)}
    depth_mm, mask = geometry(obs)
    return {"rgb": rgb, "depth_mm": depth_mm, "mask": mask, "eef20": eef20_from_observation(obs),
            "prompt": format_prompt(instruction_of(obs)), "seed": int(seed), "policy": policy or {}, "full": full}


__all__ = ["CAMERAS", "LIVE_CAMERAS", "RobotGeometry", "eef20_from_observation", "eef20_to_action_dicts",
           "instruction_of", "joint_vector", "pose_wxyz_to_arm10", "request_from_observation", "rot6d_to_quat_wxyz",
           "training_rgb"]
