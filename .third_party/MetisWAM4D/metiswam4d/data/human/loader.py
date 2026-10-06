"""Human pretraining loader: weighted mixture of raw-window datasets with modality-homogeneous batches.

Components declare their modality signature (``video`` or ``video+track``); the shared
:class:`~metiswam4d.data.mixture.MixtureBatchSampler` draws the signature sequence identically on
every rank (FSDP runs the same experts in lock-step) and fills the batch rank-specifically.  A
validation loader over the held-out IG-10K episodes is built the same way with a fixed schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from metiswam4d.data.human.ig10k import IG10KConfig, IG10KWindowDataset
from metiswam4d.data.human.kling import KlingConfig, KlingWindowDataset
from metiswam4d.data.mixture import MixtureBatchSampler, MixtureComponent, MixtureDataset


@dataclass
class HumanComponentConfig:
    name: str
    kind: str                                  # kling | ig10k
    weight: float = 1.0
    kling: KlingConfig | None = None
    ig10k: IG10KConfig | None = None

    @property
    def signature(self) -> tuple[str, ...]:
        if self.kind == "kling":
            return ("video",) if (self.kling or KlingConfig()).mode == "video" else ("video", "track")
        if (self.ig10k or IG10KConfig()).with_action:
            return ("video", "track", "action")
        return ("video", "track")


@dataclass
class HumanDataConfig:
    components: list[HumanComponentConfig] = field(default_factory=list)
    val: IG10KConfig | None = None             # held-out IG-10K human episodes (split=val) -> val/*
    val_robot: IG10KConfig | None = None       # held-out IG-10K robot episodes (profile=robot) -> val_robot/*
    val_batches: int = 8                       # batches per rank per evaluation (each split)

    def val_splits(self) -> dict[str, IG10KConfig]:
        return {name: cfg for name, cfg in (("val", self.val), ("val_robot", self.val_robot)) if cfg is not None}


def build_component(cfg: HumanComponentConfig) -> Dataset:
    if cfg.kind == "kling":
        return KlingWindowDataset(cfg.kling or KlingConfig())
    if cfg.kind == "ig10k":
        return IG10KWindowDataset(cfg.ig10k or IG10KConfig())
    raise ValueError(f"unknown human component kind {cfg.kind!r}")


def collate_human(samples: list[dict]) -> dict:
    """Stack the samples; a tensor key missing from some samples of a mixed Kling / IG-10K track batch is
    zero-filled (IG-10K has no hand meshes -> ``hand_present`` False; Kling track pixels are rendered from its
    meshes on the GPU and overwrite the zeros)."""
    batch: dict = {}
    keys = dict.fromkeys(k for s in samples for k in s)
    for key in keys:
        values = [s.get(key) for s in samples]
        if any(v is None for v in values):
            ref = next(v for v in values if v is not None)
            if not torch.is_tensor(ref):
                continue
            values = [v if v is not None else torch.zeros_like(ref) for v in values]
        if key in ("key", "prompt"):
            batch[key] = list(values)
        elif key == "embodiment":
            batch[key] = torch.as_tensor([int(v) for v in values])
        elif torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
    return batch


def build_human_loader(cfg: HumanDataConfig, *, batch_size: int, num_batches: int, num_workers: int, seed: int,
                       rank: int, world_size: int, epoch: int = 0) -> tuple[DataLoader, Sampler]:
    if not cfg.components:
        raise ValueError("data.human.components is empty")
    components = [MixtureComponent(build_component(c), c.weight, c.name, signature=c.signature) for c in cfg.components]
    mixture = MixtureDataset(components)
    sampler = MixtureBatchSampler(mixture, batch_size, num_batches, seed=seed, rank=rank, world_size=world_size)
    sampler.set_epoch(epoch)
    loader = DataLoader(mixture, batch_sampler=sampler, collate_fn=collate_human, num_workers=num_workers,
                        pin_memory=True, persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)
    return loader, sampler


class _FixedBatchSampler(Sampler[list[int]]):
    """Deterministic rank-disjoint batches for evaluation."""

    def __init__(self, length: int, batch_size: int, num_batches: int, *, seed: int, rank: int, world_size: int):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(length, generator=g)
        need = batch_size * num_batches * world_size
        if need > length:
            perm = perm.repeat((need + length - 1) // length)
        flat = perm[:need].reshape(world_size, num_batches, batch_size)[rank]
        self.batches = [b.tolist() for b in flat]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def build_human_val_loaders(cfg: HumanDataConfig, *, batch_size: int, num_workers: int, seed: int, rank: int,
                            world_size: int) -> dict[str, DataLoader]:
    """One fixed-schedule loader per held-out split (``val`` = IG-10K human, ``val_robot`` = IG-10K robot); the
    log prefix is the split name."""
    loaders = {}
    for name, split in cfg.val_splits().items():
        dataset = IG10KWindowDataset(split)
        sampler = _FixedBatchSampler(len(dataset), batch_size, cfg.val_batches, seed=seed, rank=rank, world_size=world_size)
        loaders[name] = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_human, num_workers=num_workers,
                                   pin_memory=True)
    return loaders


__all__ = ["HumanComponentConfig", "HumanDataConfig", "build_component", "build_human_loader", "build_human_val_loaders",
           "collate_human"]
