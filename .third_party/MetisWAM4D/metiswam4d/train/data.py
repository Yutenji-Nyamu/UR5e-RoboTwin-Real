"""Build the training data pipeline from a DataConfig."""
from __future__ import annotations

from torch.utils.data import DataLoader

from metiswam4d.config import DataConfig
from metiswam4d.data import (
    LatentShardDataset, MixtureBatchSampler, MixtureDataset, SyntheticLatentDataset, collate, default_registry,
)
from metiswam4d.data.mixture import MixtureComponent


def build_mixture(cfg: DataConfig) -> MixtureDataset:
    if not cfg.components:
        raise ValueError("data.components is empty")
    registry = default_registry()
    components = []
    for comp in cfg.components:
        if comp.manifest is None:
            ds = SyntheticLatentDataset(
                cfg.spec, comp.synthetic_length,
                with_video="video" in comp.modalities, with_track="track" in comp.modalities,
                with_action="action" in comp.modalities, with_roles=comp.with_roles,
                embodiment=registry.index(comp.embodiment))
        else:
            ds = LatentShardDataset(comp.manifest, cfg.spec, registry=registry,
                                    drop_modalities=comp.drop_modalities, validate=cfg.validate)
        components.append(MixtureComponent(ds, comp.weight, comp.name))
    return MixtureDataset(components)


def _raw_window_loader(dataset, cfg: DataConfig, *, seed: int, rank: int, world_size: int, epoch: int):
    """Raw episode windows (RT2 / RoboDojo), sharded per rank; the caller encodes them on the GPU."""
    from torch.utils.data import DistributedSampler
    from metiswam4d.data.rt2 import collate_raw
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)
    sampler.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate_raw,
                        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
                        drop_last=True, prefetch_factor=4 if cfg.num_workers > 0 else None)
    return loader, sampler


def build_rt2_loader(cfg: DataConfig, *, seed: int, rank: int, world_size: int, epoch: int = 0):
    from metiswam4d.data.rt2 import RT2EpisodeDataset
    return _raw_window_loader(RT2EpisodeDataset(cfg.rt2), cfg, seed=seed, rank=rank, world_size=world_size, epoch=epoch)


def build_robodojo_loader(cfg: DataConfig, *, seed: int, rank: int, world_size: int, epoch: int = 0):
    from metiswam4d.data.robodojo import window_dataset
    return _raw_window_loader(window_dataset(cfg.robodojo), cfg, seed=seed, rank=rank, world_size=world_size,
                              epoch=epoch)


def build_loader(cfg: DataConfig, *, num_batches: int, seed: int, rank: int, world_size: int,
                 epoch: int = 0):
    if cfg.rt2 is not None:
        return build_rt2_loader(cfg, seed=seed, rank=rank, world_size=world_size, epoch=epoch)
    if cfg.robodojo is not None:
        return build_robodojo_loader(cfg, seed=seed, rank=rank, world_size=world_size, epoch=epoch)
    if cfg.human is not None:
        from metiswam4d.data.human import build_human_loader
        return build_human_loader(cfg.human, batch_size=cfg.batch_size, num_batches=num_batches,
                                  num_workers=cfg.num_workers, seed=seed, rank=rank, world_size=world_size, epoch=epoch)
    mixture = build_mixture(cfg)
    sampler = MixtureBatchSampler(mixture, cfg.batch_size, num_batches, seed=seed, rank=rank, world_size=world_size)
    sampler.set_epoch(epoch)
    loader = DataLoader(mixture, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0)
    return loader, sampler


def build_val_loaders(cfg: DataConfig, *, seed: int, rank: int, world_size: int) -> dict:
    """Held-out evaluation loaders keyed by log prefix (human / robot IG-10K splits, RoboDojo ``val``); empty
    without validation."""
    if cfg.robodojo is not None:
        if not cfg.robodojo.val_split:
            return {}
        from dataclasses import replace
        from metiswam4d.data.human.loader import _FixedBatchSampler
        from metiswam4d.data.robodojo import window_dataset
        from metiswam4d.data.rt2 import collate_raw
        dataset = window_dataset(replace(cfg.robodojo, split=cfg.robodojo.val_split, samples_per_episode=1,
                                         fixed_windows=True))
        sampler = _FixedBatchSampler(len(dataset), cfg.batch_size, cfg.robodojo.val_batches, seed=seed, rank=rank,
                                     world_size=world_size)
        return {"val": DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_raw,
                                  num_workers=min(cfg.num_workers, 4), pin_memory=True)}
    if cfg.human is None:
        return {}
    from metiswam4d.data.human.loader import build_human_val_loaders
    return build_human_val_loaders(cfg.human, batch_size=cfg.batch_size, num_workers=min(cfg.num_workers, 4),
                                   seed=seed, rank=rank, world_size=world_size)


__all__ = ["build_loader", "build_mixture", "build_val_loaders"]
