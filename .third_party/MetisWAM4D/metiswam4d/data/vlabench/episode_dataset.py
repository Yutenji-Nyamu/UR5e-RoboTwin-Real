"""VLABench training windows (``scripts/vlabench/generate_4d.py`` episodes) in the RT2 raw-window contract.

    video   frames s + 4k, k = 0..8 -> L layout 384x320: ``forward`` (head) 256x320 top, wrist camera bottom-left,
            ``right`` camera bottom-right, LANCZOS from 480x480 (the OpenWAM VLABench reader / eval client)
    track   head pixels (du px, dv px, dd m) s+4k -> s+4k+4, k = 0..7 on the 240x320 stretched head grid; robot
            (arm + gripper) and movable objects are foreground, static geometry is known-static (zero motion)
    action  ``actions7`` rows s .. s+31 -> EEF10 (xyz, rot6d, gripper) -> Alpha VLABench min-max -> unified slots
            0-9; proprio ``state7[s]`` through the same map (OpenWAM ``VLABenchDataset``)
    text    the episode instruction in the Alpha deploy template, UMT5 features from ``<root>/text_cache``
"""
from __future__ import annotations

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

from metiswam4d.data.rt2.codec import SHRINK_D_M, anchor_frame, encode_uvd
from metiswam4d.data.rt2.eef import load_pickled_stats
from metiswam4d.data.rt2.episode_dataset import ACTION_STEPS, SOURCE_FRAMES, STRIDE, TRACK_ROWS, VIDEO_FRAMES
from metiswam4d.data.rt2.text_cache import PromptTextCache, format_prompt

EEF10 = 10
UNIFY_DIM = 80
CAMERAS = ("head_rgb", "left_rgb", "right_rgb")


def euler7_to_eef10(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    mat = Rotation.from_euler("xyz", np.asarray(x[..., 3:6], np.float64)).as_matrix()
    rot6d = np.concatenate((mat[..., :, 0], mat[..., :, 1]), axis=-1)
    return np.concatenate((x[..., :3], rot6d, x[..., 6:7]), axis=-1).astype(np.float32)


def lshape_layout(views: list[Image.Image], out_h: int, out_w: int) -> np.ndarray:
    top_h = int(round(out_h * 2.0 / 3.0))
    half_w = out_w // 2
    canvas = Image.new("RGB", (out_w, out_h))
    canvas.paste(views[0].resize((out_w, top_h), Image.LANCZOS), (0, 0))
    canvas.paste(views[1].resize((half_w, out_h - top_h), Image.LANCZOS), (0, top_h))
    canvas.paste(views[2].resize((out_w - half_w, out_h - top_h), Image.LANCZOS), (half_w, top_h))
    return np.asarray(canvas, dtype=np.uint8)


def read_index(root: Path, index_file: str, split: str, tasks) -> list[dict]:
    rows = []
    with open(root / index_file) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split and (tasks is None or row["task"] in tasks) and row["frames"] >= SOURCE_FRAMES:
                    rows.append(row)
    if not rows:
        raise ValueError(f"no complete episodes for split {split!r} in {root / index_file}")
    return rows


class VLABenchEpisodeDataset(Dataset):
    """Map-style over episodes x samples_per_episode; the window start is drawn per item."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config.root)
        self.rows = read_index(self.root, config.index_file, config.split, config.tasks)
        eef = load_pickled_stats(config.alpha_stats)["eef"]
        self.lo = np.asarray(eef["min"], np.float32)[:EEF10]
        self.range = np.maximum(np.asarray(eef["max"], np.float32)[:EEF10] - self.lo, 1e-6)
        self.text = PromptTextCache(self.root / "text_cache")

    def __len__(self) -> int:
        return len(self.rows) * self.config.samples_per_episode

    def __getitem__(self, index: int) -> dict:
        """A corrupt episode must not kill a multi-day run: the failure is recorded in
        ``<root>/_logs/bad_episodes.jsonl`` and a different window is returned."""
        for attempt in range(8):
            try:
                return self._load(index)
            except (OSError, KeyError, ValueError, IndexError) as exc:
                self._record_bad(self.rows[index // self.config.samples_per_episode], exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("VLABenchEpisodeDataset: 8 consecutive unreadable episodes")

    def _record_bad(self, row: dict, exc: Exception) -> None:
        try:
            log_dir = self.root / "_logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "bad_episodes.jsonl", "a") as f:
                f.write(json.dumps({"task": row["task"], "episode": row["episode"], "error": repr(exc)[:200],
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[vlabench dataset] unreadable {row['task']}/episode{row['episode']}: {exc!r}", flush=True)

    def _rng(self, index: int) -> random.Random:
        if self.config.fixed_windows:
            return random.Random(hash((self.config.seed, index)) & 0xFFFFFFFF)
        return random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)

    def _normalize(self, euler7: np.ndarray) -> np.ndarray:
        eef = euler7_to_eef10(euler7)
        out = np.zeros((*eef.shape[:-1], UNIFY_DIM), np.float32)
        out[..., :EEF10] = 2.0 * (eef - self.lo) / self.range - 1.0
        return out

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = self._rng(index)
        with h5py.File(self.root / row["task"] / f"episode{row['episode']}.h5") as h:
            n = int(h.attrs["frames"])
            s = rng.randint(0, n - SOURCE_FRAMES)
            video_idx = [s + STRIDE * k for k in range(VIDEO_FRAMES)]
            track_idx = video_idx[:TRACK_ROWS]
            frames = np.stack([
                lshape_layout([Image.open(io.BytesIO(h[c][t].tobytes())).convert("RGB") for c in CAMERAS],
                              self.config.layout_height, self.config.layout_width)
                for t in video_idx])                                                        # [9, 384, 320, 3]
            head = Image.open(io.BytesIO(h["head_rgb"][s].tobytes())).convert("RGB")
            head_depth_mm = np.asarray(h["head_depth_mm"][s], np.float32)
            role_rows = np.stack([h["role"][t] for t in track_idx]).astype(np.uint8)      # [8, 240, 320]
            uvd = np.stack([h["delta_uvd"][t] for t in track_idx]).astype(np.float32)     # [8, 240, 320, 3]
            actions7 = np.asarray(h["actions7"][s:s + ACTION_STEPS], np.float32)
            state7 = np.asarray(h["state7"][s:s + 1], np.float32)
            instruction = str(h.attrs["instruction"])
        if actions7.shape[0] != ACTION_STEPS:
            raise ValueError(f"short actions in {row['task']}/episode{row['episode']}: {actions7.shape}")

        h_grid, w_grid = role_rows.shape[-2:]
        foreground = role_rows > 0
        robot_anchor = role_rows[0] == 1
        track_px, track_delta = encode_uvd(uvd, foreground, w_grid, SHRINK_D_M["robodojo"])
        track_rgb = np.concatenate((anchor_frame(robot_anchor)[None], track_px), axis=0)
        action = self._normalize(actions7)
        dims = np.zeros(UNIFY_DIM, bool)
        dims[:EEF10] = True
        prompt = format_prompt(instruction)
        progress = torch.tensor([s / max(n - 1, 1)], dtype=torch.float32)
        return {
            "progress_video": progress,
            "progress_body": progress.clone(),
            "video_frames": torch.from_numpy(frames),
            "track_rgb": torch.from_numpy(track_rgb),
            "track_foreground": torch.from_numpy(np.concatenate((robot_anchor[None], foreground))),
            "track_role_px": torch.from_numpy(np.concatenate((robot_anchor[None].astype(np.uint8), role_rows))),
            "track_delta": torch.from_numpy(track_delta).to(torch.float16),
            "track_unknown_role": torch.tensor(False),
            "head_rgb": torch.from_numpy(np.array(head.resize((w_grid, h_grid), Image.LANCZOS), dtype=np.uint8)),
            "head_depth_mm": torch.from_numpy(head_depth_mm),
            "head_mask": torch.from_numpy(robot_anchor),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(np.broadcast_to(dims, (ACTION_STEPS, UNIFY_DIM)).copy()),
            "proprio": torch.from_numpy(self._normalize(state7)),
            "proprio_mask": torch.from_numpy(dims[None].copy()),
            "text_context": self.text.load(prompt),
            "embodiment": self.config.embodiment,
            "key": f"{row['task']}/episode{row['episode']}/s{s}",
            "prompt": prompt,
        }


__all__ = ["VLABenchEpisodeDataset", "euler7_to_eef10", "lshape_layout"]
