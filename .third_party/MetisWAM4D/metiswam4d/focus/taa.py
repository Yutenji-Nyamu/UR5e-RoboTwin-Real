"""Transition-Anchored Aggregation (TAA).

Per query, the ``N_f`` interaction tokens along the frame-level axis are
aggregated into one token whose temporal centre is anchored on the predicted
coupling-transition profile ``s_n``: ordered anchors are the inverse of the
normalised cumulative transition strength at levels ``(k - 1/2) / K``
(differentiable piecewise-linear inverse), refined by learned offsets and
window widths.  Aggregation weights combine a content score with a time penalty
around the anchor (Gaussian, or Gaussian + learned residual profile).  With a
flat ``s_n`` the anchors fall back to uniform sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


def inverse_cumulative(strength: Tensor, levels: Tensor, eps: float = 1e-6) -> Tensor:
    """Differentiable ``F^{-1}(levels)`` for per-frame strengths ``[B, N]``.

    ``F`` is the normalised cumulative sum on knots ``u_n = n / N``; ``levels``
    ``[K]`` in (0, 1).  Returns anchors ``[B, K]`` in [0, 1].  The strengths are
    scaled to unit mean before ``eps`` is added, so ``eps`` is relative: a small
    but structured profile keeps its anchors, only a flat profile gives uniform ones.
    """
    b, n = strength.shape
    mass = strength.clamp(min=0)
    mass = mass / mass.mean(dim=1, keepdim=True).clamp(min=1e-30) + eps  # all-zero rows stay flat
    cdf = torch.cat((torch.zeros(b, 1, dtype=mass.dtype, device=mass.device),
                     mass.cumsum(dim=1) / mass.sum(dim=1, keepdim=True)), dim=1)  # [B, N+1]
    knots = torch.arange(n + 1, dtype=mass.dtype, device=mass.device) / n
    with torch.no_grad():
        idx = torch.searchsorted(cdf, levels[None].expand(b, -1).contiguous()).clamp(1, n)
    lo, hi = cdf.gather(1, idx - 1), cdf.gather(1, idx)
    frac = (levels[None] - lo) / (hi - lo + eps)
    return (knots[idx - 1] + frac.clamp(0, 1) / n).clamp(0, 1)


@dataclass
class Anchors:
    centre: Tensor      # [B, K]  c_k
    width: Tensor       # [B, K]  w_k
    raw: Tensor         # [B, K]  c_bar_k before offsets


class TAA(nn.Module):
    """``window="gaussian"``: fixed Gaussian penalty around the anchor (learned width only).
    ``window="adaptive"``: Gaussian plus a zero-initialised residual profile predicted from the signed
    offset, the width and a per-query embedding, so the window can become asymmetric (e.g. emphasise
    frames after a transition), heavier-tailed or multi-modal; at initialisation both are identical."""

    def __init__(self, d: int, num_queries: int, d_cond: int, w_min: float = 0.02,
                 w_max: float = 0.5, eps: float = 1e-6, window: str = "gaussian", refine: bool = False,
                 fixed_width: float = 0.15, max_offset: float = 0.25):
        """``refine=False`` (default): anchors are exactly the transition quantiles and the window width is
        ``fixed_width`` (normalised time; 0.15 ~ 1.2 of 8 slots) -- nothing to drift.  ``refine=True``:
        learned offsets bounded to ``+-max_offset`` and learned widths in ``[w_min, w_max]``."""
        super().__init__()
        if not 0 < w_min < w_max:
            raise ValueError("require 0 < w_min < w_max")
        if window not in ("gaussian", "adaptive"):
            raise ValueError(f"unknown temporal window {window!r}")
        self.d, self.K = d, num_queries
        self.w_min, self.w_max, self.eps = w_min, w_max, eps
        self.window = window
        self.use_refine, self.fixed_width, self.max_offset = refine, fixed_width, max_offset
        self.register_buffer("levels", (torch.arange(num_queries) + 0.5) / num_queries, persistent=False)
        if refine:
            self.refine = nn.Linear(d_cond + d, 2 * num_queries)
            nn.init.zeros_(self.refine.weight)
            nn.init.zeros_(self.refine.bias)
        self.time_key = nn.Linear(d, d)
        if window == "adaptive":
            hidden = 32
            self.query_shape = nn.Parameter(torch.randn(num_queries, hidden) * 0.02)
            self.profile = nn.Sequential(nn.Linear(3 + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.profile[-1].weight)
            nn.init.zeros_(self.profile[-1].bias)

    def penalty(self, u: Tensor, anchors: Anchors) -> Tensor:
        """Non-negative-ish time penalty ``[B, K, N]`` subtracted from the content score."""
        du = u[None, None] - anchors.centre[..., None]              # [B, K, N]
        width = anchors.width[..., None]
        gauss = du ** 2 / (2 * width ** 2)
        if self.window == "gaussian":
            return gauss
        b, k, n = du.shape
        feats = torch.stack((du, du / width, width.expand_as(du)), dim=-1)  # [B, K, N, 3]
        shape = self.query_shape[None, :, None].expand(b, k, n, -1).to(feats.dtype)
        residual = self.profile(torch.cat((feats, shape), dim=-1)).squeeze(-1)
        return gauss + residual

    def anchors(self, frame_strength: Tensor, cond: Tensor, pooled: Tensor) -> Anchors:
        """``frame_strength [B, N]``, ``cond [B, d_cond]``, ``pooled [B, d]``."""
        raw = inverse_cumulative(frame_strength.float(), self.levels.float(), self.eps).to(cond.dtype)
        if not self.use_refine:
            width = torch.full_like(raw, self.fixed_width)
            return Anchors(raw, width, raw)
        delta, b = self.refine(torch.cat((cond, pooled), dim=-1)).chunk(2, dim=-1)
        centre = (raw + self.max_offset * torch.tanh(delta)).clamp(0.0, 1.0)
        width = self.w_min + (self.w_max - self.w_min) * torch.sigmoid(b)
        return Anchors(centre, width, raw)

    def aggregate(self, tokens: Tensor, queries: Tensor, anchors: Anchors) -> tuple[Tensor, Tensor]:
        """``tokens [B, K, N, d]``, ``queries [B, K, d]`` -> ``(z [B, K, d], beta [B, K, N])``."""
        b, k, n, d = tokens.shape
        u = (torch.arange(n, dtype=tokens.dtype, device=tokens.device) + 0.5) / n
        content = torch.einsum("bkd,bknd->bkn", queries, self.time_key(tokens)) / math.sqrt(d)
        beta = (content - self.penalty(u, anchors)).softmax(dim=-1)
        return torch.einsum("bkn,bknd->bkd", beta, tokens), beta


__all__ = ["Anchors", "TAA", "inverse_cumulative"]
