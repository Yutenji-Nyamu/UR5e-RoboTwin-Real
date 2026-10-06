"""Raw RT2 training windows (pixels + actions + cached text); VAE encoding happens on the GPU.

Window (OpenWAM-Alpha contract): start ``s``, 33 source frames.
    video   frames s + 4k, k = 0..8   -> 3-view L-shaped layout 384x320 (head 256x320 top, wrists 128x160)
    track   transitions s+4k -> s+4k+4, k = 0..7, plus the zero-displacement anchor at s   (head camera 240x320);
            pixel-frame (du/W, dv/W, dd) from the stored camera-frame displacement, GT depth and intrinsics
    action  frames s+1 .. s+32 (EEF20 -> unified 80-D), proprio at s
    text    one prompt sampled from the episode's instruction pool, UMT5 features from the cache
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import time
from typing import Sequence

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from metiswam4d.data.rt2.codec import SHRINK_D_M, anchor_frame, encode_uvd, uvd_from_xyz
from metiswam4d.data.rt2.eef import EEFNormalizer, read_eef20, scatter_unified
from metiswam4d.data.rt2.text_cache import PromptTextCache, load_instruction_index

SOURCE_FRAMES = 33
STRIDE = 4
VIDEO_FRAMES = 9
TRACK_ROWS = 8
ACTION_STEPS = 32


@dataclass
class RT2WindowConfig:
    root: str = "/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D"
    index_file: str = "index.jsonl"
    variants: tuple[str, ...] = ("demo_clean_4d", "demo_randomized_4d")
    tasks: tuple[str, ...] | None = None
    alpha_stats: str = "/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full/normalization_stats.npy"
    samples_per_episode: int = 4
    layout_height: int = 384
    layout_width: int = 320
    cameras: tuple[str, str, str] = ("head_camera", "left_camera", "right_camera")
    embodiment: int = 2  # registry index of robotwin2_aloha_agilex
    seed: int = 0


def read_index(root: Path, index_file: str, variants: Sequence[str], tasks: Sequence[str] | None) -> list[dict]:
    rows = []
    with open(root / index_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["variant"] in variants and (tasks is None or row["task"] in tasks) and row["frames"] >= SOURCE_FRAMES:
                rows.append(row)
    if not rows:
        raise ValueError(f"no complete episodes for variants {variants} in {root / index_file}")
    return rows


def decode_jpeg(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(bytes(raw))).convert("RGB")


def multiview_layout(frames: dict[str, Image.Image], cameras: Sequence[str], out_h: int, out_w: int) -> np.ndarray:
    """OpenWAM/FastWAM L-shape: top camera 2/3 height full width, two bottom cameras half width each."""
    top_h = int(round(out_h * 2.0 / 3.0))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    canvas = Image.new("RGB", (out_w, out_h))
    canvas.paste(frames[cameras[0]].resize((out_w, top_h), Image.BILINEAR), (0, 0))
    canvas.paste(frames[cameras[1]].resize((half_w, bottom_h), Image.BILINEAR), (0, top_h))
    canvas.paste(frames[cameras[2]].resize((out_w - half_w, bottom_h), Image.BILINEAR), (half_w, top_h))
    return np.asarray(canvas, dtype=np.uint8)


class RT2EpisodeDataset(Dataset):
    """Map-style over episodes x samples_per_episode; the window start is drawn per item."""

    def __init__(self, config: RT2WindowConfig):
        self.config = config
        self.root = Path(config.root)
        self.rows = read_index(self.root, config.index_file, config.variants, config.tasks)
        self.eef = EEFNormalizer(config.alpha_stats)
        self.prompts = load_instruction_index(self.root / "episode_instructions_sim_aligned.jsonl")
        self.text = PromptTextCache(self.root / "text_cache")

    def __len__(self) -> int:
        return len(self.rows) * self.config.samples_per_episode

    def episode_dir(self, row: dict) -> Path:
        return self.root / row["task"] / row["variant"] / f"episode{row['episode']}"

    def __getitem__(self, index: int) -> dict:
        """A corrupt episode (e.g. HDF5 chunks lost in a node reboot) must not kill a multi-day run: the
        failure is recorded in ``<root>/_logs/bad_episodes.jsonl`` and a different window is returned."""
        for attempt in range(8):
            try:
                return self._load(index)
            except (OSError, KeyError, ValueError) as exc:
                row = self.rows[index // self.config.samples_per_episode]
                self._record_bad(row, exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("RT2EpisodeDataset: 8 consecutive unreadable episodes")

    def _record_bad(self, row: dict, exc: Exception) -> None:
        try:
            log_dir = self.root / "_logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "bad_episodes.jsonl", "a") as f:
                f.write(json.dumps({"task": row["task"], "variant": row["variant"], "episode": row["episode"],
                                    "error": repr(exc)[:200], "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[rt2 dataset] unreadable {row['task']}/{row['variant']}/episode{row['episode']}: {exc!r}", flush=True)

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)
        n = int(row["frames"])
        s = rng.randint(0, n - SOURCE_FRAMES)
        d = self.episode_dir(row)
        cams = self.config.cameras
        video_idx = [s + STRIDE * k for k in range(VIDEO_FRAMES)]

        with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
            frames = np.stack([
                multiview_layout({c: decode_jpeg(src[f"observation/{c}/rgb"][t]) for c in cams}, cams,
                                 self.config.layout_height, self.config.layout_width)
                for t in video_idx])                                                  # [9, 384, 320, 3]
            head_rgb = np.asarray(decode_jpeg(src["observation/head_camera/rgb"][s]), dtype=np.uint8)
            depth_mm = np.stack([src["observation/head_camera/depth"][t] for t in video_idx[:-1]]).astype(np.float32)  # [8, H, W]
            K = np.asarray(src["observation/head_camera/intrinsic_cv"][s], dtype=np.float64)
            role_rows = np.stack([t4d["role"][t] for t in video_idx[:-1]])           # [8, H, W]  frame s+4k
            delta = np.stack([t4d["delta_xyz_cam"][t] for t in video_idx[:-1]]).astype(np.float32)  # [8, H, W, 3]
            valid = np.stack([t4d["valid"][t] for t in video_idx[:-1]]) > 0          # [8, H, W]
            eef = read_eef20(src, np.arange(s, s + ACTION_STEPS + 1))                 # [33, 20]: proprio + 32 actions

        head_depth_mm = depth_mm[0]
        foreground = valid & (role_rows > 0)
        uvd = uvd_from_xyz(delta, depth_mm / 1000.0, K)                               # [8, H, W, 3] (px, px, m)
        track_px, track_delta = encode_uvd(uvd, foreground, foreground.shape[-1], SHRINK_D_M["rt2"])
        track_rgb = np.concatenate((anchor_frame(role_rows[0] > 0)[None], track_px), axis=0)  # [9, H, W, 3]
        normalized = self.eef.normalize(eef)
        unified, dims = scatter_unified(normalized)
        action_mask = np.broadcast_to(dims, (ACTION_STEPS, unified.shape[-1])).copy()

        key = f"{row['task']}/{row['variant']}/episode{row['episode']}"
        pool = self.prompts.get(key) or [""]
        prompt = rng.choice(pool)
        text = self.text.load(prompt)
        progress = torch.tensor([s / max(n - 1, 1)], dtype=torch.float32)  # current frame / episode length

        return {
            "progress_video": progress,                                                  # aligned streams:
            "progress_body": progress.clone(),                                           # same phase
            "video_frames": torch.from_numpy(frames),                                   # uint8 [9, 384, 320, 3]
            "track_rgb": torch.from_numpy(track_rgb),                                   # uint8 [9, 240, 320, 3]
            "track_foreground": torch.from_numpy(np.concatenate((role_rows[:1] > 0, foreground))),  # bool [9, H, W]
            "track_role_px": torch.from_numpy(np.concatenate((role_rows[:1], role_rows)).astype(np.uint8)),  # [9, H, W]
            "track_delta": torch.from_numpy(track_delta).to(torch.float16),            # [8, H, W, 3] normalised (u, v, d)
            "head_rgb": torch.from_numpy(head_rgb),
            "head_depth_mm": torch.from_numpy(head_depth_mm),
            "head_mask": torch.from_numpy(role_rows[0] > 0),
            "action": torch.from_numpy(unified[1:]),                                     # [32, 80]
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(unified[:1]),                                    # [1, 80]
            "proprio_mask": torch.from_numpy(dims[None].copy()),
            "text_context": text,                                                        # bf16 [L, 4096]
            "embodiment": self.config.embodiment,
            "key": f"{key}/s{s}",
            "prompt": prompt,
        }


def collate_raw(samples: list[dict]) -> dict:
    batch: dict = {}
    lengths = [s["text_context"].shape[0] for s in samples]
    text = torch.zeros(len(samples), max(lengths), samples[0]["text_context"].shape[1], dtype=torch.bfloat16)
    mask = torch.zeros(len(samples), max(lengths), dtype=torch.bool)
    for i, s in enumerate(samples):
        text[i, :lengths[i]] = s["text_context"]
        mask[i, :lengths[i]] = True
    batch["text_context"], batch["text_mask"] = text, mask
    for key in samples[0]:
        if key in ("text_context", "key", "prompt", "embodiment"):
            continue
        batch[key] = torch.stack([s[key] for s in samples])
    batch["key"] = [s["key"] for s in samples]
    batch["prompt"] = [s["prompt"] for s in samples]
    batch["embodiment"] = torch.as_tensor([int(s["embodiment"]) for s in samples])
    return batch


__all__ = ["ACTION_STEPS", "RT2EpisodeDataset", "RT2WindowConfig", "STRIDE", "collate_raw", "multiview_layout", "read_index"]
