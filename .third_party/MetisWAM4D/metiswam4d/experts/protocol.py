"""Common interface every expert exposes to the mixture-of-transformers driver.

An expert owns its input embedding, time embedding, Transformer blocks and
output head.  The driver only needs four hooks per macro layer:

    state = expert.prepare(**inputs)
    q, k, v, post = expert.pre_attn(layer, state)     # Q/K/V in the shared attention space
    state = expert.post_attn(layer, state, attn, post)
    output = expert.finalize(state)

Q/K/V are ``[B, S, heads * head_dim]`` so that the streams of different
experts can be concatenated into one joint softmax.  Experts shallower than the
macro depth (the Track expert) update only at mapped layers and expose the last
K/V otherwise (``updates_at``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


def macro_layer_map(expert_layers: int, macro_layers: int) -> tuple[int, ...]:
    """Macro layer index at which each expert block runs (evenly spread)."""
    if expert_layers < 1 or macro_layers < expert_layers:
        raise ValueError("macro depth must be at least the expert depth")
    if expert_layers == 1:
        return (macro_layers - 1,)
    result = tuple(
        round(index * (macro_layers - 1) / (expert_layers - 1)) for index in range(expert_layers)
    )
    if len(set(result)) != expert_layers:
        raise RuntimeError("layer mapping produced duplicate macro layers")
    return result


@dataclass
class ExpertState:
    """Mutable per-forward state of one expert."""

    tokens: Tensor
    modulation: Tensor
    head_time: Tensor
    rope: Tensor | None = None
    context: Tensor | None = None
    context_mask: Tensor | None = None
    clean_count: int = 0
    grid: tuple[int, int, int] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def shallow_copy(self, tokens: Tensor) -> "ExpertState":
        """Copy with replaced tokens; used inside activation checkpointing."""
        return ExpertState(
            tokens=tokens, modulation=self.modulation, head_time=self.head_time,
            rope=self.rope, context=self.context, context_mask=self.context_mask,
            clean_count=self.clean_count, grid=self.grid, extras=dict(self.extras),
        )


class ExpertAdapter(nn.Module):
    """Abstract base; concrete experts implement the four hooks."""

    name: str = "expert"

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def num_heads(self) -> int:
        raise NotImplementedError

    @property
    def head_dim(self) -> int:
        raise NotImplementedError

    @property
    def hidden_dim(self) -> int:
        raise NotImplementedError

    @property
    def attention_width(self) -> int:
        return self.num_heads * self.head_dim

    def block_index(self, layer: int) -> int | None:
        """Own block that runs at macro layer ``layer`` (None: reuse cached K/V)."""
        return layer if layer < self.num_layers else None

    def updates_at(self, layer: int) -> bool:
        return self.block_index(layer) is not None

    def prepare(self, **inputs: Any) -> ExpertState:
        raise NotImplementedError

    def pre_attn(self, layer: int, state: ExpertState) -> tuple[Tensor, Tensor, Tensor, Any]:
        raise NotImplementedError

    def post_attn(self, layer: int, state: ExpertState, attn_out: Tensor, post: Any) -> ExpertState:
        raise NotImplementedError

    def finalize(self, state: ExpertState) -> Tensor:
        raise NotImplementedError

    def hidden(self, state: ExpertState) -> Tensor:
        """Current hidden tokens ``[B, S, hidden_dim]`` (read by the focus module)."""
        return state.tokens

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameter groups for the optimizer; default puts everything in ``name``."""
        return {self.name: [p for p in self.parameters() if p.requires_grad]}


def split_heads(x: Tensor, heads: int) -> Tensor:
    b, s, width = x.shape
    return x.reshape(b, s, heads, width // heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    b, h, s, d = x.shape
    return x.transpose(1, 2).reshape(b, s, h * d)


__all__ = ["ExpertAdapter", "ExpertState", "macro_layer_map", "merge_heads", "split_heads"]
