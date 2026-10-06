"""Raw RoboDojo (RDJ_MetisWAM4D) training windows; VAE encoding happens on the GPU (``RT2OnlineEncoder``).

Same window contract as RT2 (OpenWAM-Alpha): start ``s``, 33 source frames.
    video   frames s + 4k, k = 0..8   -> 3-view L-shaped layout 384x320 (head 256x320 top, wrists 128x160)
    track   robot-pixel (du px, dv px, dd m) transitions s+4k -> s+4k+4, k = 0..7 (``track4d.h5/delta_uvd``), plus
            the zero-displacement anchor at s; pixels with ``role == 0`` are *unknown* (objects, background) and
            are black / excluded from every Track target
    action  frames s+1 .. s+32, EEF20 world -> per-arm base frame -> Alpha RoboDojo min-max -> unified 80-D;
            proprio at s
    text    the task instruction (one per task), UMT5 features from the cache

Source JPEGs are BGR-ordered: PIL's "RGB" decode must be channel-reversed.  The head depth is rendered for the
robot pixels only (millimetres, 0 elsewhere) and is used as-is for the depth condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import time

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from metiswam4d.data.robodojo.eef_base import world_to_base_eef20
from metiswam4d.data.rt2.codec import SHRINK_D_M, anchor_frame, encode_uvd
from metiswam4d.data.rt2.eef import EEFNormalizer, scatter_unified
from metiswam4d.data.rt2.episode_dataset import (
    ACTION_STEPS, SOURCE_FRAMES, STRIDE, TRACK_ROWS, VIDEO_FRAMES, multiview_layout,
)
from metiswam4d.data.rt2.text_cache import PromptTextCache, format_prompt


@dataclass
class RDJWindowConfig:
    source: str = "rdj"                        # rdj | ebench (EBench LeRobot buckets) | vlabench | vlabench_mix
    official_root: str | None = None           # vlabench_mix: the official VLABench LeRobot release (video + action)
    root: str = "/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D"
    index_file: str = "index.jsonl"
    split: str = "train"                       # train | val (index.jsonl ``split`` column)
    tasks: tuple[str, ...] | None = None
    alpha_stats: str = ("/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/"
                        "OpenWAM-Alpha-Sim-RoboDojo/normalization_stats.npy")
    samples_per_episode: int = 4
    fixed_windows: bool = False                # window start depends only on (seed, index): evaluation / panels
    layout_height: int = 384
    layout_width: int = 320
    cameras: tuple[str, str, str] = ("head_camera", "left_camera", "right_camera")
    embodiment: int = 1                        # registry index of robodojo_arx_x5
    seed: int = 0
    val_split: str | None = "val"              # held-out split evaluated every ``training.eval_every`` steps
    val_batches: int = 4                       # batches per rank per evaluation


def read_index(root: Path, index_file: str, split: str, tasks) -> list[dict]:
    rows = []
    with open(root / index_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] == split and (tasks is None or row["task"] in tasks) and row["frames"] >= SOURCE_FRAMES:
                rows.append(row)
    if not rows:
        raise ValueError(f"no complete episodes for split {split!r} in {root / index_file}")
    return rows


def load_official_instructions(path: str | Path) -> dict[str, list[str]]:
    """``episode_instructions_official.jsonl`` -> {"<task>/<variant>/episodeN": [formatted prompts]}."""
    index: dict[str, list[str]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parts = Path(row["source"]).parts   # .../<task>/<variant>/data/episodeN.hdf5
            task, variant, episode = parts[-4], parts[-3], Path(parts[-1]).stem
            index[f"{task}/{variant}/{episode}"] = [format_prompt(p) for p in row.get("instructions") or []]
    return index


def decode_bgr_jpeg(raw: bytes) -> Image.Image:
    """The stored JPEGs were encoded from BGR arrays: reverse the channels to get RGB."""
    arr = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))[..., ::-1]
    return Image.fromarray(np.ascontiguousarray(arr))


class RDJEpisodeDataset(Dataset):
    """Map-style over episodes x samples_per_episode; the window start is drawn per item."""

    def __init__(self, config: RDJWindowConfig):
        self.config = config
        self.root = Path(config.root)
        self.rows = read_index(self.root, config.index_file, config.split, config.tasks)
        self.eef = EEFNormalizer(config.alpha_stats)
        self.prompts = load_official_instructions(self.root / "episode_instructions_official.jsonl")
        self.text = PromptTextCache(self.root / "text_cache")

    def __len__(self) -> int:
        return len(self.rows) * self.config.samples_per_episode

    def episode_dir(self, row: dict) -> Path:
        return self.root / row["task"] / row["variant"] / f"episode{row['episode']}"

    def __getitem__(self, index: int) -> dict:
        """A corrupt episode must not kill a multi-day run: the failure is recorded in
        ``<root>/_logs/bad_episodes.jsonl`` and a different window is returned."""
        for attempt in range(8):
            try:
                return self._load(index)
            except (OSError, KeyError, ValueError) as exc:
                row = self.rows[index // self.config.samples_per_episode]
                self._record_bad(row, exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("RDJEpisodeDataset: 8 consecutive unreadable episodes")

    def _record_bad(self, row: dict, exc: Exception) -> None:
        try:
            log_dir = self.root / "_logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "bad_episodes.jsonl", "a") as f:
                f.write(json.dumps({"task": row["task"], "variant": row["variant"], "episode": row["episode"],
                                    "error": repr(exc)[:200], "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[rdj dataset] unreadable {row['task']}/{row['variant']}/episode{row['episode']}: {exc!r}", flush=True)

    def _rng(self, index: int) -> random.Random:
        if self.config.fixed_windows:
            return random.Random(hash((self.config.seed, index)) & 0xFFFFFFFF)
        return random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = self._rng(index)
        n = int(row["frames"])
        s = rng.randint(0, n - SOURCE_FRAMES)
        d = self.episode_dir(row)
        cams = self.config.cameras
        video_idx = [s + STRIDE * k for k in range(VIDEO_FRAMES)]
        track_idx = video_idx[:TRACK_ROWS]

        with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
            unknown_background = not str(t4d.attrs.get("schema", "")).endswith(".objects.v2")
            frames = np.stack([
                multiview_layout({c: decode_bgr_jpeg(src[f"observation/{c}/rgb"][t]) for c in cams}, cams,
                                 self.config.layout_height, self.config.layout_width)
                for t in video_idx])                                                  # [9, 384, 320, 3]
            head_rgb = np.array(decode_bgr_jpeg(src["observation/head_camera/rgb"][s]), dtype=np.uint8)
            head_depth_mm = np.asarray(src["observation/head_camera/depth"][s], dtype=np.float32)  # robot pixels, mm
            role_rows = np.stack([t4d["role"][t] for t in track_idx]).astype(np.uint8)  # [8, H, W]  frame s+4k
            uvd = np.stack([t4d["delta_uvd"][t] for t in track_idx]).astype(np.float32)  # [8, H, W, 3] (px, px, m)
            eef_world = np.asarray(t4d["eef20"][s:s + ACTION_STEPS + 1], dtype=np.float32)  # [33, 20]
        if eef_world.shape[0] != ACTION_STEPS + 1:
            raise ValueError(f"short eef20 in {d}: {eef_world.shape}")
        if not np.isfinite(eef_world).all():
            raise ValueError(f"non-finite eef20 in {d} at s={s}")

        foreground = role_rows > 0                                                       # supervised robot/object targets
        robot_anchor = role_rows[0] == 1                                                 # deployment inputs stay robot-only
        track_px, track_delta = encode_uvd(uvd, foreground, foreground.shape[-1], SHRINK_D_M["robodojo"])
        track_rgb = np.concatenate((anchor_frame(robot_anchor)[None], track_px), axis=0)  # [9, H, W, 3]
        normalized = self.eef.normalize(world_to_base_eef20(eef_world))
        unified, dims = scatter_unified(normalized)
        action_mask = np.broadcast_to(dims, (ACTION_STEPS, unified.shape[-1])).copy()

        key = f"{row['task']}/{row['variant']}/episode{row['episode']}"
        pool = self.prompts.get(key) or [""]
        prompt = rng.choice(pool)
        text = self.text.load(prompt)
        progress = torch.tensor([s / max(n - 1, 1)], dtype=torch.float32)

        return {
            "progress_video": progress,
            "progress_body": progress.clone(),
            "video_frames": torch.from_numpy(frames),                                   # uint8 [9, 384, 320, 3]
            "track_rgb": torch.from_numpy(track_rgb),                                   # uint8 [9, 240, 320, 3]
            "track_foreground": torch.from_numpy(np.concatenate((robot_anchor[None], foreground))),
            "track_role_px": torch.from_numpy(np.concatenate((robot_anchor[None].astype(np.uint8), role_rows))),
            "track_delta": torch.from_numpy(track_delta).to(torch.float16),            # [8, H, W, 3] normalised (u, v, d)
            "track_unknown_role": torch.tensor(unknown_background),
            "head_rgb": torch.from_numpy(head_rgb),
            "head_depth_mm": torch.from_numpy(head_depth_mm),
            "head_mask": torch.from_numpy(robot_anchor),
            "action": torch.from_numpy(unified[1:]),                                     # [32, 80]
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(unified[:1]),                                    # [1, 80]
            "proprio_mask": torch.from_numpy(dims[None].copy()),
            "text_context": text,                                                        # bf16 [L, 4096]
            "embodiment": self.config.embodiment,
            "key": f"{key}/s{s}",
            "prompt": prompt,
        }


__all__ = ["RDJEpisodeDataset", "RDJWindowConfig", "decode_bgr_jpeg", "load_official_instructions", "read_index"]
