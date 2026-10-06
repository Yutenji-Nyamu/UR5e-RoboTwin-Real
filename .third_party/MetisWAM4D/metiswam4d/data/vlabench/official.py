"""The official VLABench primitive fine-tune release (``vlabench_primitive_ft_lerobot_video``, the data OpenWAM-Alpha
VLABench was trained on and the same scene version as the frozen evaluation tracks) as raw training windows.

Decoding and action conversion are OpenWAM's own ``VLABenchDataset`` built with Alpha's normalization statistics, so
pixels and actions are exactly what the Alpha checkpoint saw: L layout 384x320 (``image`` / ``wrist_image`` /
``second_image``), EEF10 actions rows s .. s+31 and proprio row s, min-max into unified slots 0-9.  The release has no
depth, masks or object poses: Track pixels are all unknown (no Track target), the depth / mask conditions are empty, and
the head condition image is the canvas head slot resized to 240x320.  The prompt is the instruction in the training
template (the release uses the same 128 instructions as ``gen4d``, all in its text cache).
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

from metiswam4d.data.rt2.eef import load_pickled_stats
from metiswam4d.data.rt2.episode_dataset import SOURCE_FRAMES
from metiswam4d.data.rt2.text_cache import PromptTextCache, format_prompt

TRACK_HEIGHT, TRACK_WIDTH = 240, 320


class OfficialVLABenchDataset(Dataset):
    def __init__(self, config, root: str | Path, text_root: str | Path):
        from openwam.dataloader.vlabench import VLABenchDataset
        self.config = config
        self.root = Path(root)
        stats = load_pickled_stats(config.alpha_stats)["eef"]
        self.reader = VLABenchDataset(str(self.root), split="train", num_frames=SOURCE_FRAMES, video_stride=4,
                                      window_stride=1, height=384, width=320, multiview=True,
                                      normalize_mode="min-max", normalization_stats_path=config.alpha_stats,
                                      action_stats=stats, unify_action=True, unify_action_map=["0-9"],
                                      color_jitter=None)
        valid = self.reader._ep_valid_end - self.reader._ep_valid_start          # full 33-frame windows only
        episodes = self.reader._eps_df["episode_index"].to_numpy()
        self.rows = [{"local": e, "episode": int(episodes[e]), "frames": int(valid[e])}
                     for e in range(len(valid)) if valid[e] >= SOURCE_FRAMES]
        self.text = PromptTextCache(Path(text_root))

    def __len__(self) -> int:
        return len(self.rows) * self.config.samples_per_episode

    def __getitem__(self, index: int) -> dict:
        for attempt in range(8):
            try:
                return self._load(index)
            except (OSError, KeyError, ValueError, IndexError, RuntimeError) as exc:
                self._record_bad(self.rows[index // self.config.samples_per_episode], exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("OfficialVLABenchDataset: 8 consecutive unreadable episodes")

    def _record_bad(self, row: dict, exc: Exception) -> None:
        try:
            log_dir = self.root / "_logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "bad_episodes.jsonl", "a") as f:
                f.write(json.dumps({"episode": row["episode"], "error": repr(exc)[:200],
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[official vlabench] unreadable episode {row['episode']}: {exc!r}", flush=True)

    def _rng(self, index: int) -> random.Random:
        if self.config.fixed_windows:
            return random.Random(hash((self.config.seed, index)) & 0xFFFFFFFF)
        return random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = self._rng(index)
        n = row["frames"]
        s = rng.randint(0, n - SOURCE_FRAMES)
        out = self.reader._getitem_impl(int(self.reader._cum_n_starts[row["local"]]) + s)
        frames = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in out["video"]])  # [9, 384, 320, 3]
        head = Image.fromarray(frames[0, :256]).resize((TRACK_WIDTH, TRACK_HEIGHT), Image.LANCZOS)
        empty = np.zeros((9, TRACK_HEIGHT, TRACK_WIDTH), dtype=bool)
        prompt = format_prompt(out["prompt"])
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
            "action": out["action"].float()[:32],
            "action_mask": out["action_mask"][:32].bool(),
            "proprio": out["proprio"].float().reshape(1, -1),
            "proprio_mask": out["proprio_mask"].reshape(1, -1).bool(),
            "text_context": self.text.load(prompt),
            "embodiment": self.config.embodiment,
            "key": f"official/episode{row['episode']}/s{s}",
            "prompt": prompt,
        }


class VLABenchMixDataset(Dataset):
    """``gen4d`` episodes (Track, depth, masks) followed by the official release (video + action only); both use
    ``samples_per_episode``, so each source contributes in proportion to its episode count.  Validation is ``gen4d``
    only (the official release has no held-out split)."""

    def __init__(self, config):
        from metiswam4d.data.vlabench.episode_dataset import VLABenchEpisodeDataset
        self.config = config
        self.parts = [VLABenchEpisodeDataset(config)]
        if config.split == "train":
            self.parts.append(OfficialVLABenchDataset(config, config.official_root, Path(config.root) / "text_cache"))
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])
        self.rows = [r for p in self.parts for r in p.rows]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> dict:
        part = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.parts[part][index - int(self.offsets[part])]


__all__ = ["OfficialVLABenchDataset", "VLABenchMixDataset"]
