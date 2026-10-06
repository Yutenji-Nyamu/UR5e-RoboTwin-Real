"""Per-group learning-rate curriculum that protects pretrained priors.

Each parameter group (video / track / action / focus / proprio) has a target
learning rate, an optional freeze period (lr = 0) and a linear ramp.  A global
constant or cosine schedule multiplies all groups.  Stage recipes:

    stage 1  Video frozen while the interpolated Track and the reader warm up,
             then Video ramps to a very small lr (3e-7).
    stage 2  Action (from Alpha) frozen while the compact reading projections
             and the reader adapt, then ramps to 1e-6; Video / Track continue
             at low lr.
    stage 3  same groups per domain; ``compact`` vs ``dense`` comparisons share
             the recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from metiswam4d.config import GroupLR, TrainingConfig


def prior_ramp_factor(step: int, freeze_steps: int, ramp_steps: int) -> float:
    """0 during the freeze, linear to 1 over the ramp, 1 afterwards."""
    if freeze_steps < 0 or ramp_steps < 0:
        raise ValueError("freeze/ramp steps must be non-negative")
    if step < freeze_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, (step - freeze_steps) / ramp_steps)


def global_factor(step: int, cfg: TrainingConfig) -> float:
    if cfg.lr_schedule == "constant":
        return 1.0
    if cfg.lr_schedule == "cosine":
        start = int(cfg.lr_schedule_start)
        progress = min(1.0, max(0, step - start) / max(1, cfg.max_steps - start))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr schedule {cfg.lr_schedule!r}")


@dataclass
class GroupState:
    name: str
    spec: GroupLR
    params: list[nn.Parameter]

    def lr_at(self, step: int, cfg: TrainingConfig) -> float:
        return self.spec.lr * prior_ramp_factor(step, self.spec.freeze_steps, self.spec.ramp_steps) * global_factor(step, cfg)


class Curriculum:
    """Owns the optimizer parameter groups and updates their learning rates."""

    def __init__(self, groups: dict[str, list[nn.Parameter]], cfg: TrainingConfig):
        self.cfg = cfg
        self.groups: list[GroupState] = []
        for name, params in groups.items():
            if name not in cfg.groups:
                raise KeyError(f"no learning-rate spec for parameter group {name!r}")
            self.groups.append(GroupState(name, cfg.groups[name], params))
        unused = set(cfg.groups) - set(groups)
        self.unused_groups = sorted(unused)

    def build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = []
        for g in self.groups:
            param_groups.append({
                "params": g.params, "lr": g.lr_at(0, self.cfg), "name": g.name,
                "weight_decay": self.cfg.weight_decay if g.spec.weight_decay is None else g.spec.weight_decay,
            })
        return torch.optim.AdamW(param_groups, betas=tuple(self.cfg.betas), eps=1e-8)

    def step(self, optimizer: torch.optim.Optimizer, step: int) -> dict[str, float]:
        lrs = {}
        for g, param_group in zip(self.groups, optimizer.param_groups):
            lr = g.lr_at(step, self.cfg)
            param_group["lr"] = lr
            lrs[f"lr/{g.name}"] = lr
        return lrs

    def frozen_groups(self, step: int) -> list[str]:
        return [g.name for g in self.groups if g.lr_at(step, self.cfg) == 0.0]


__all__ = ["Curriculum", "GroupState", "global_factor", "prior_ramp_factor"]
