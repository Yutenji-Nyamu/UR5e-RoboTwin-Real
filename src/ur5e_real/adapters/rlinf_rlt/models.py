# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (see integrations/rlt/LICENSE-RLinf).
# Adapted from the source-locked RLTMLPPolicy/MLPPolicy/QHead for single-device UR5.
"""Same three-layer tanh actor / LayerNorm twin-Q; only physical dimensions differ."""

from __future__ import annotations

import math
import torch
from torch import nn


def mlp(inputs, outputs, width, layers, *, critic=False):
    modules = []
    for i in range(layers + 1):
        final = i == layers
        layer = nn.Linear(inputs, outputs if final else width)
        if critic and final:
            nn.init.normal_(layer.weight, std=0.02)
        else:
            gain = (5 / 3) if critic else (0.01 * math.sqrt(2) if final else math.sqrt(2))
            nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
        modules.append(layer)
        if not final:
            if critic:
                modules.append(nn.LayerNorm(width))
            modules.append(nn.Tanh())
        inputs = width
    return nn.Sequential(*modules)


class RLTActor(nn.Module):
    def __init__(self, z_dim, proprio_dim=13, chunk_len=20, width=256, layers=3, fixed_std=0.002):
        super().__init__()
        self.chunk_len, self.fixed_std = chunk_len, fixed_std
        self.net = mlp(z_dim + proprio_dim + chunk_len * 7, chunk_len * 7, width, layers)

    def forward(self, z, proprio, reference, *, deterministic=False, reference_dropout=0.0):
        ref = reference.flatten(1)
        if reference_dropout:
            keep = torch.rand((ref.shape[0], 1), device=ref.device) >= reference_dropout
            ref = ref * keep
        raw = self.net(torch.cat([ref, z, proprio], dim=-1))
        if not deterministic:
            raw = raw + torch.randn_like(raw) * self.fixed_std
        return torch.tanh(raw).reshape(-1, self.chunk_len, 7)


class TwinQ(nn.Module):
    def __init__(self, z_dim, proprio_dim=13, chunk_len=20, width=256, layers=3):
        super().__init__()
        self.qs = nn.ModuleList(
            [mlp(z_dim + proprio_dim + chunk_len * 7, 1, width, layers, critic=True) for _ in range(2)]
        )

    def forward(self, z, proprio, actions):
        state_action = torch.cat([z, proprio, actions.flatten(1)], dim=-1)
        return torch.cat([q(state_action) for q in self.qs], dim=-1)


def heads(config):
    algorithm = config["algorithm"]
    options = {
        "z_dim": config["token"]["embed_dim"],
        "proprio_dim": algorithm["proprio_dim"],
        "chunk_len": config["real"]["action_steps"],
        "width": algorithm["hidden_dim"],
        "layers": algorithm["hidden_layers"],
    }
    return RLTActor(**options, fixed_std=algorithm["fixed_std"]), TwinQ(**options)
