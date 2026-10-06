"""EBench training windows in the RoboDojo / RT2 raw-window contract, encoded online by ``RT2OnlineEncoder``.

Decoding and action conversion are OpenWAM's own ``EBenchDataset`` (the reader OpenWAM-Alpha-Sim-EBench was trained
with), built with that checkpoint's normalization statistics, so pixels and actions match the Alpha initialization
exactly:
    video   frames s + 4k, k = 0..8 -> 3-view L layout 384x320 (overlook 256x320 top, wrists 128x160)
    action  raw-23 EEF (per-arm xyz + rot6d + gripper, base dx / dy / dyaw deg) frames s .. s+31, min-max,
            scattered to the unified 80-D slots 0-9 / 34-43 / 68-70; proprio at s
    text    the episode instruction in the Alpha deploy template, UMT5 features from ``<root>/text_cache``

EBench has no depth, masks or camera poses, and the demos cannot be replayed (no object poses), so this source
carries no Track: every Track pixel has role 0 (unknown) and no foreground, which excludes it from all Track
targets, and the depth / mask conditions are empty.  Held out per task bucket: its last ``VAL_EPISODES`` episodes.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from metiswam4d.data.rt2.episode_dataset import SOURCE_FRAMES
from metiswam4d.data.rt2.text_cache import PromptTextCache

VAL_EPISODES = 2
TRACK_HEIGHT, TRACK_WIDTH = 240, 320
CAMERAS = ("video.overlook_camera_view", "video.left_camera_view", "video.right_camera_view")


def _buckets(root: Path, alpha_stats: str) -> list:
    from openwam.dataloader.ebench import EBenchDataset, discover_ebench_buckets
    stats = np.load(alpha_stats, allow_pickle=True).item()["eef"]
    return [EBenchDataset(str(b), dataset_id=str(b.relative_to(root)), split="train", num_frames=SOURCE_FRAMES,
                          video_stride=4, window_stride=1, height=384, width=320, multiview=True,
                          target_camera=CAMERAS[0], camera_layout=list(CAMERAS), normalize_mode="min-max",
                          normalization_stats_path=alpha_stats, action_stats=stats, unify_action=True,
                          color_jitter=None)
            for b in discover_ebench_buckets(str(root))]


class EBenchEpisodeDataset(Dataset):
    """Map-style over episodes x samples_per_episode; the window start is drawn per item (full 33-frame windows)."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config.root)
        self.buckets = _buckets(self.root, config.alpha_stats)
        self.rows = []
        for b, bucket in enumerate(self.buckets):
            episodes = bucket._episodes
            for e, ep in enumerate(episodes):
                split = "val" if e >= len(episodes) - VAL_EPISODES else "train"
                if split == config.split and int(ep["length"]) >= SOURCE_FRAMES:
                    task = bucket._dataset_id
                    if config.tasks is None or task in config.tasks:
                        self.rows.append({"bucket": b, "local": e, "task": task, "variant": "demo",
                                          "episode": int(ep["episode_index"]), "frames": int(ep["length"]),
                                          "split": split})
        if not self.rows:
            raise ValueError(f"no complete EBench episodes for split {config.split!r} under {self.root}")
        self.text = PromptTextCache(self.root / "text_cache")

    def __len__(self) -> int:
        return len(self.rows) * self.config.samples_per_episode

    def __getitem__(self, index: int) -> dict:
        """A corrupt episode must not kill a multi-day run: the failure is recorded in
        ``<root>/_logs/bad_episodes.jsonl`` and a different window is returned."""
        for attempt in range(8):
            try:
                return self._load(index)
            except (OSError, KeyError, ValueError, RuntimeError) as exc:
                self._record_bad(self.rows[index // self.config.samples_per_episode], exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("EBenchEpisodeDataset: 8 consecutive unreadable episodes")

    def _record_bad(self, row: dict, exc: Exception) -> None:
        try:
            log_dir = self.root / "_logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "bad_episodes.jsonl", "a") as f:
                f.write(json.dumps({"task": row["task"], "episode": row["episode"], "error": repr(exc)[:200],
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[ebench dataset] unreadable {row['task']}/episode{row['episode']}: {exc!r}", flush=True)

    def _rng(self, index: int) -> random.Random:
        if self.config.fixed_windows:
            return random.Random(hash((self.config.seed, index)) & 0xFFFFFFFF)
        return random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = self._rng(index)
        n = row["frames"]
        s = rng.randint(0, n - SOURCE_FRAMES)
        bucket = self.buckets[row["bucket"]]
        out = bucket._getitem_impl(int(bucket._cum_n_starts[row["local"]]) + s)

        frames = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in out["video"]])  # [9, 384, 320, 3]
        head = Image.fromarray(frames[0, :256]).resize((TRACK_WIDTH, TRACK_HEIGHT), Image.BICUBIC)
        action = out["action"].float()
        proprio = out["proprio"].float().reshape(1, -1)
        empty = np.zeros((9, TRACK_HEIGHT, TRACK_WIDTH), dtype=bool)
        text = self.text.load(out["prompt"])
        progress = torch.tensor([s / max(n - 1, 1)], dtype=torch.float32)
        return {
            "progress_video": progress,
            "progress_body": progress.clone(),
            "video_frames": torch.from_numpy(frames),
            "track_rgb": torch.zeros(9, TRACK_HEIGHT, TRACK_WIDTH, 3, dtype=torch.uint8),
            "track_foreground": torch.from_numpy(empty),
            "track_role_px": torch.zeros(9, TRACK_HEIGHT, TRACK_WIDTH, dtype=torch.uint8),
            "track_delta": torch.zeros(8, TRACK_HEIGHT, TRACK_WIDTH, 3, dtype=torch.float16),
            "track_unknown_role": torch.tensor(True),
            "head_rgb": torch.from_numpy(np.array(head, dtype=np.uint8)),
            "head_depth_mm": torch.zeros(TRACK_HEIGHT, TRACK_WIDTH, dtype=torch.float32),
            "head_mask": torch.from_numpy(empty[0]),
            "action": action[:32],
            "action_mask": out["action_mask"][:32].bool(),
            "proprio": proprio,
            "proprio_mask": out["proprio_mask"].reshape(1, -1).bool(),
            "text_context": text,
            "embodiment": self.config.embodiment,
            "key": f"{row['task']}/episode{row['episode']}/s{s}",
            "prompt": out["prompt"],
        }


__all__ = ["EBenchEpisodeDataset", "VAL_EPISODES"]
