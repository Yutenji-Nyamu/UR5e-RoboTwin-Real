"""Weighted mixtures of datasets with modality-homogeneous batches.

Batches must contain samples with identical modality presence (see
``contract.collate``).  ``MixtureBatchSampler`` first picks a *group* of
datasets sharing a presence signature according to the summed weights, then
fills the batch from datasets inside the group according to their weights.
It is rank-aware for distributed training (disjoint deterministic streams).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from metiswam4d.data.contract import presence


@dataclass
class MixtureComponent:
    dataset: Dataset
    weight: float
    name: str = ""
    signature: tuple[str, ...] | None = None  # inferred from the first sample when None


class MixtureDataset(Dataset):
    """Concatenation with bookkeeping of component offsets."""

    def __init__(self, components: Sequence[MixtureComponent]):
        if not components:
            raise ValueError("mixture needs at least one component")
        self.components = list(components)
        for c in self.components:
            if c.weight < 0:
                raise ValueError("weights must be non-negative")
            if c.signature is None:
                c.signature = presence(c.dataset[0])
        self.offsets = []
        total = 0
        for c in self.components:
            self.offsets.append(total)
            total += len(c.dataset)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict:
        for offset, c in zip(reversed(self.offsets), reversed(self.components)):
            if index >= offset:
                return c.dataset[index - offset]
        raise IndexError(index)


class MixtureBatchSampler(Sampler[list[int]]):
    def __init__(self, mixture: MixtureDataset, batch_size: int, num_batches: int, *,
                 seed: int = 0, rank: int = 0, world_size: int = 1):
        self.mixture, self.batch_size, self.num_batches = mixture, batch_size, num_batches
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, c in enumerate(mixture.components):
            if c.weight > 0 and len(c.dataset) > 0:
                groups.setdefault(c.signature, []).append(i)
        if not groups:
            raise ValueError("no component with positive weight")
        self.groups = list(groups.values())
        self.group_weights = torch.tensor(
            [sum(mixture.components[i].weight for i in g) for g in self.groups], dtype=torch.float64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        # The group (modality signature) sequence is shared by all ranks: under FSDP every rank must
        # run the same experts in lock-step.  Item picks inside the group are rank-specific.
        g_group = torch.Generator().manual_seed(self.seed * 7919 + self.epoch * 104729)
        g = torch.Generator().manual_seed(self.seed * 7919 + self.epoch * 104729 + 1 + self.rank)
        for _ in range(self.num_batches):
            group = self.groups[int(torch.multinomial(self.group_weights / self.group_weights.sum(), 1, generator=g_group))]
            weights = torch.tensor([self.mixture.components[i].weight for i in group], dtype=torch.float64)
            picks = torch.multinomial(weights / weights.sum(), self.batch_size, replacement=True, generator=g)
            batch = []
            for pick in picks.tolist():
                comp = group[pick]
                n = len(self.mixture.components[comp].dataset)
                batch.append(self.mixture.offsets[comp] + int(torch.randint(n, (1,), generator=g)))
            yield batch


__all__ = ["MixtureBatchSampler", "MixtureComponent", "MixtureDataset"]
