"""Manifest-driven dataset over precomputed latent samples.

A manifest is a JSONL file; each line describes one sample::

    {"path": "shards/000123.pt", "key": "robot_H10_L0/ep7/s40", "embodiment": "ig10k_realman"}

``path`` (relative to the manifest directory unless absolute) points to a
``torch.save``d dict with contract keys, or an ``.npz`` archive with the same
keys.  Modalities are inferred from the stored keys; ``drop_modalities`` lets a
stage ignore stored blocks (e.g. IG-10K Cross: keep Video from the human
demonstration shard but Track/Action from the robot shard is done offline).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from metiswam4d.data.contract import SampleSpec, validate_sample
from metiswam4d.data.embodiments import EmbodimentRegistry, default_registry


class LatentShardDataset(Dataset):
    def __init__(self, manifest: str | Path, spec: SampleSpec, *, registry: EmbodimentRegistry | None = None,
                 drop_modalities: Sequence[str] = (), validate: bool = True, limit: int | None = None):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.spec = spec
        self.registry = registry or default_registry()
        self.drop = tuple(drop_modalities)
        self.validate = validate
        with open(self.manifest) as handle:
            self.entries = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            self.entries = self.entries[:limit]
        if not self.entries:
            raise ValueError(f"empty manifest {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, path: Path) -> dict:
        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu", weights_only=True)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                data = {k: torch.from_numpy(archive[k]) for k in archive.files}
        else:
            raise ValueError(f"unsupported shard format {path.suffix}")
        return dict(data)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = self.root / path
        sample = self._load(path)
        for modality in self.drop:
            keys = {"video": ("video_clean",),
                    "track": ("track_clean", "rgb_condition", "depth_condition", "mask_condition",
                              "track_valid", "track_role", "track_disp_frames", "track_role_frames"),
                    "action": ("action", "action_mask", "proprio", "proprio_mask")}[modality]
            for key in keys:
                sample.pop(key, None)
        sample["key"] = entry.get("key", str(index))
        name = entry.get("embodiment", "human_ego")
        sample["embodiment"] = self.registry.index(name)
        if self.validate:
            validate_sample(sample, self.spec)
        return sample


__all__ = ["LatentShardDataset"]
