"""Sub-frame temporal unfolding of VAE-compressed latent frames.

Each future latent frame ``j`` (``J`` of them) folds ``kappa`` physical frames.
Unfolding maps its hidden feature to ``kappa`` slot features

    h_hat[n, i] = W_tau h[j, i] + p_tau,      n = kappa * (j - 1) + tau,

so that frame-level quantities (per-frame displacement, roles, coupling
transitions) can be located on a frame-level time axis ``n = 1..N_f`` that is
shared with the Track displacement rows and the Action steps.  A light
decoding head ``g`` regresses the token-grid displacement of slot ``n`` and a
role head classifies background / body / object; both are supervised so that
slot ``tau`` really corresponds to physical frame ``tau``.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SubFrameUnfold(nn.Module):
    """``[B, J, h, w, d_in] -> [B, J * kappa, h, w, d_out]``."""

    def __init__(self, d_in: int, d_out: int, kappa: int):
        super().__init__()
        self.kappa = kappa
        self.d_out = d_out
        self.norm = nn.LayerNorm(d_in)
        self.slots = nn.Linear(d_in, kappa * d_out)
        self.slot_position = nn.Parameter(torch.zeros(kappa, d_out))
        nn.init.normal_(self.slot_position, std=0.02)

    def forward(self, hidden: Tensor) -> Tensor:
        b, j, h, w, _ = hidden.shape
        x = self.slots(self.norm(hidden)).view(b, j, h, w, self.kappa, self.d_out)
        x = x + self.slot_position
        return x.permute(0, 1, 4, 2, 3, 5).reshape(b, j * self.kappa, h, w, self.d_out)


class DisplacementHead(nn.Module):
    """``g``: slot feature -> 3-D displacement in normalised metric units."""

    def __init__(self, d_in: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_in
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RoleHead(nn.Module):
    """Slot feature -> logits over (background, body, object)."""

    def __init__(self, d_in: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_in
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def frames_to_grid(tokens: Tensor, grid: tuple[int, int, int], skip_frames: int = 1) -> Tensor:
    """``[B, F*h*w, d] -> [B, F - skip, h, w, d]`` dropping the leading clean frames."""
    b, s, d = tokens.shape
    f, h, w = grid
    if s != f * h * w:
        raise ValueError(f"token count {s} does not match grid {grid}")
    return tokens.view(b, f, h, w, d)[:, skip_frames:]


__all__ = ["DisplacementHead", "RoleHead", "SubFrameUnfold", "frames_to_grid"]
