"""Training windows for the Memory / Open model: RDJ-format episodes, InternW0 canvas, sparse memory, joint actions.

Window at start ``s`` (25 Hz source frames, the RT2 / RDJ contract otherwise unchanged):

    memory   frames [0, s-128, s-96, s-64, s-32] (episode frame 0 + the first frames of the previous four 32-step
             chunk windows), each clamped to >= 0, so early windows repeat frame 0; in training each history slot
             is independently replaced by frame 0 with ``memory_drop`` (inference may have fewer real frames)
    video    frames s + 4k, k = 0..8, on InternW0's canvas: 384x256, head 256x256 on top, left / right wrists
             128x128 below; each camera is first brought to 240x320, then resized to its cell (bilinear, antialias)
    track    unchanged: head camera robot / object pixel uvd transitions (role 1 robot, 2 object, 3 gripper ->
             robot); role 0 is unknown in robot-only episodes
    action   official ``joint_action/vector`` at s .. s+31 (14-D absolute joints), proprio ``joint_state/vector`` at
             s, both z-scored with InternW0's statistics into the unified 80-D slots
    text     the episode instruction in the RoboTwin / InternW0 video prompt template, UMT5 features from the cache

Source JPEGs are BGR-ordered (``decode_bgr_jpeg``); the canvas is RGB, the same byte order Isaac returns at
evaluation time.  Official demonstrations (pre-2026-09-15 release) carry a one-frame RGB lag: their colour frames are
read at index + 1, so every image matches the joint state it is paired with, as at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from metiswam4d.data.robodojo.episode_dataset import RDJEpisodeDataset, RDJWindowConfig, decode_bgr_jpeg
from metiswam4d.data.rt2 import RT2OnlineEncoder
from metiswam4d.data.rt2.codec import SHRINK_D_M, anchor_frame, encode_uvd
from metiswam4d.data.rt2.episode_dataset import ACTION_STEPS, SOURCE_FRAMES, STRIDE, TRACK_ROWS, VIDEO_FRAMES
from metiswam4d.data.rt2.text_cache import PromptTextCache, format_prompt

from metiswam4d_inspired_by_internw0.joint_action import JointNormalizer, scatter

CANVAS_H, CANVAS_W = 384, 256
CAMERA_HW = (240, 320)
CHUNK = 32
MEMORY_CHUNKS = 4
MEMORY_SLOTS = 1 + MEMORY_CHUNKS
ROLE_GRIPPER = 3
OFFICIAL_RGB_SHIFT = 1


def memory_indices(s: int) -> list[int]:
    """Episode frame 0, then the starts of the previous ``MEMORY_CHUNKS`` chunk windows, oldest first."""
    return [0] + [max(s - CHUNK * k, 0) for k in range(MEMORY_CHUNKS, 0, -1)]


def _resize(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x, size=hw, mode="bilinear", antialias=True, align_corners=False)


def internw0_canvas(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB cameras -> uint8 ``[384, 256, 3]`` canvas (InternW0 ``build_training_exact_canvas``)."""
    cams = [_resize(torch.from_numpy(np.array(c, dtype=np.uint8)).permute(2, 0, 1)[None].float().div(255.0), CAMERA_HW)
            for c in (head, left, right)]
    top_h, left_w = (CANVAS_H * 2) // 3, CANVAS_W // 2
    bottom_h = CANVAS_H - top_h
    top = _resize(cams[0], (top_h, CANVAS_W))
    bottom = torch.cat([_resize(cams[1], (bottom_h, left_w)), _resize(cams[2], (bottom_h, CANVAS_W - left_w))], dim=-1)
    canvas = torch.cat([top, bottom], dim=-2)[0].clamp(0, 1)
    return canvas.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


@dataclass
class IW0Extras:
    memory_drop: float = 0.15
    text_cache: str | None = None          # UMT5 cache of the templated prompts (default: <root>/text_cache)
    instructions: str = "instructions.jsonl"       # {"key": "<task>/<variant>/episodeN", "instructions": [...]}


class IW0EpisodeDataset(RDJEpisodeDataset):
    def __init__(self, config: RDJWindowConfig, extras: IW0Extras | None = None):
        self.config = config
        self.extras = extras or IW0Extras()
        self.root = Path(config.root)
        from metiswam4d.data.robodojo.episode_dataset import read_index
        self.rows = read_index(self.root, config.index_file, config.split, config.tasks)
        self.joints = JointNormalizer()
        self.prompts: dict[str, list[str]] = {}
        with open(self.root / self.extras.instructions) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.prompts[row["key"]] = [format_prompt(t) for t in row["instructions"]]
        self.text = PromptTextCache(Path(self.extras.text_cache or self.root / "text_cache"))

    def canvas(self, src: h5py.File, t: int) -> np.ndarray:
        cams = [np.asarray(decode_bgr_jpeg(src[f"observation/{c}/rgb"][t]), dtype=np.uint8) for c in self.config.cameras]
        return internw0_canvas(*cams)

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = self._rng(index)
        n = int(row["frames"])
        # Official release before the 2026-09-15 fix: RGB frame t shows joint state t-1 (measured on RDJ: robot mask
        # agreement 0.88 with state t-1 vs 0.53 with state t); its colour index is shifted by one.  Teacher rollouts
        # recorded in the evaluation client are aligned.
        shift = OFFICIAL_RGB_SHIFT if row["variant"] == "train_4d" else 0
        s = rng.randint(0, n - SOURCE_FRAMES - shift)
        d = self.episode_dir(row)
        video_idx = [s + STRIDE * k for k in range(VIDEO_FRAMES)]
        track_idx = video_idx[:TRACK_ROWS]
        memory_idx = memory_indices(s)
        if not self.config.fixed_windows:
            memory_idx = [0] + [0 if rng.random() < self.extras.memory_drop else t for t in memory_idx[1:]]

        with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
            unknown_background = not str(t4d.attrs.get("schema", "")).endswith((".objects.v2", ".gt4d.v1"))
            frames = np.stack([self.canvas(src, t + shift) for t in video_idx])            # [9, 384, 256, 3]
            memory = np.stack([self.canvas(src, t + shift) for t in memory_idx])           # [5, 384, 256, 3]
            head_rgb = np.array(decode_bgr_jpeg(src["observation/head_camera/rgb"][s + shift]), dtype=np.uint8)
            head_depth_mm = np.asarray(src["observation/head_camera/depth"][s], dtype=np.float32)
            role_rows = np.stack([t4d["role"][t] for t in track_idx]).astype(np.uint8)
            uvd = np.stack([t4d["delta_uvd"][t] for t in track_idx]).astype(np.float32)
            actions = np.asarray(src["joint_action/vector"][s:s + ACTION_STEPS], dtype=np.float32)   # [32, 14]
            state = np.asarray(src["joint_state/vector"][s], dtype=np.float32)                       # [14]
        if actions.shape[0] != ACTION_STEPS:
            raise ValueError(f"short joint actions in {d}: {actions.shape}")
        if not (np.isfinite(actions).all() and np.isfinite(state).all()):
            raise ValueError(f"non-finite joints in {d} at s={s}")

        role_rows[role_rows == ROLE_GRIPPER] = 1
        foreground = role_rows > 0
        robot_anchor = role_rows[0] == 1
        track_px, track_delta = encode_uvd(uvd, foreground, foreground.shape[-1], SHRINK_D_M["robodojo"])
        track_rgb = np.concatenate((anchor_frame(robot_anchor)[None], track_px), axis=0)
        action80, dims = scatter(self.joints.normalize(actions, "action"))
        proprio80, _ = scatter(self.joints.normalize(state, "state")[None])
        action_mask = np.broadcast_to(dims, action80.shape).copy()

        key = f"{row['task']}/{row['variant']}/episode{row['episode']}"
        prompt = rng.choice(self.prompts[key])
        text = self.text.load(prompt)
        progress = torch.tensor([s / max(n - 1, 1)], dtype=torch.float32)
        return {
            "progress_video": progress,
            "progress_body": progress.clone(),
            "video_frames": torch.from_numpy(frames),
            "video_memory": torch.from_numpy(memory),
            "track_rgb": torch.from_numpy(track_rgb),
            "track_foreground": torch.from_numpy(np.concatenate((robot_anchor[None], foreground))),
            "track_role_px": torch.from_numpy(np.concatenate((robot_anchor[None].astype(np.uint8), role_rows))),
            "track_delta": torch.from_numpy(track_delta).to(torch.float16),
            "track_unknown_role": torch.tensor(unknown_background),
            "head_rgb": torch.from_numpy(head_rgb),
            "head_depth_mm": torch.from_numpy(head_depth_mm),
            "head_mask": torch.from_numpy(robot_anchor),
            "action": torch.from_numpy(action80),
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(proprio80),
            "proprio_mask": torch.from_numpy(dims[None].copy()),
            "text_context": text,
            "embodiment": self.config.embodiment,
            "key": f"{key}/s{s}",
            "prompt": prompt,
        }


class IW0OnlineEncoder(RT2OnlineEncoder):
    """``RT2OnlineEncoder`` + the memory frames: each encoded as a single-frame clip (the causal Wan VAE encodes a
    first frame on its own) and prepended to the window latents -> ``video_clean [B, 48, 5 + 9, 24, 16]``."""

    @torch.no_grad()
    def __call__(self, raw: dict) -> dict:
        out = super().__call__(raw)
        memory = raw["video_memory"]
        b, k = memory.shape[:2]
        lat = self.encode_pixels(memory.reshape(b * k, 1, *memory.shape[2:]))           # [B*K, 48, 1, h, w]
        lat = lat.reshape(b, k, *lat.shape[1:]).squeeze(3).permute(0, 2, 1, 3, 4)        # [B, 48, K, h, w]
        out["video_clean"] = torch.cat((lat, out["video_clean"]), dim=2)
        return out


__all__ = ["CANVAS_H", "CANVAS_W", "IW0EpisodeDataset", "IW0Extras", "IW0OnlineEncoder", "MEMORY_SLOTS",
           "internw0_canvas", "memory_indices"]
