"""Body-object coupling field and interaction saliency.

Given frame-level token-grid displacements ``D[n, i]`` (from the unfolding head
``g``) and role probabilities ``rho[n, i, r]``:

    D_body(i)  = sum_{i'} w(i,i') rho_body(i') D(i') / (sum w rho_body + eps)   body reference motion
    r          = D - D_body                                                   coupling vector
    c          = |r| / (|D| + |D_body| + eps)  in [0, 1]                       coupling state
    dc_n       = c_n - c_{n-1}                                                coupling transition
    pi         = sum_{i'} w(i,i') rho_body(i')                                body proximity
    phi        = phi([c, |dc|, pi, |D_body|, cam]; task, state)               log-saliency (unbounded)
    s          = softmax_i(log rho_object + phi)                              object-side saliency, per frame
    s_body     = normalise_i( sum_i w(i, i') rho_object(i) s(i) )             propagated to the body side
    s_n        = sum_i rho_object exp(phi - max phi)                          frame transition strength

``w`` is a normalised neighbourhood kernel on the token grid
(:class:`SpatialKernel`: fixed Gaussian, one learned kernel, or per-token
kernels predicted from the sub-frame features).  ``phi`` is a small
FiLM-conditioned MLP whose inputs are only the physical quantities, so it can
re-weight coupling states per task but cannot read appearance; with the
adaptive kernel, appearance can only decide *which neighbours* form the
reference, never the saliency value directly.

Every consumer sees a scale-free quantity: ``s`` is a distribution over the
grid and ``s_n`` is invariant to a constant shift of ``phi`` (its scale is
normalised again in :func:`metiswam4d.focus.taa.inverse_cumulative`).  A
constant offset of ``phi`` -- the cheapest thing the FiLM can learn -- therefore
has no effect, and there is no "all-off" state in which the log-bias becomes
flat and its gradient vanishes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def gaussian_kernel(size: int, sigma: float) -> Tensor:
    if size % 2 == 0 or size < 1:
        raise ValueError("kernel size must be odd and positive")
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    return k / k.sum()


class SpatialKernel(nn.Module):
    """Normalised neighbourhood aggregation over ``[B*, C, h, w]`` maps.

    The neighbourhood weights ``w(i, i')`` define the body reference motion,
    the proximity and the body-side saliency propagation.  Three shapes:

    * ``gaussian``  fixed Gaussian (strong, hand-set prior);
    * ``learned``   one learnable ``k x k`` non-negative kernel (softmax over
                    offsets), Gaussian-initialised, translation invariant;
    * ``adaptive``  per-token kernels predicted from the sub-frame features
                    (dynamic filter, softmax over offsets, Gaussian-initialised
                    bias and zero-initialised weight).  Weights stay a convex
                    combination, so the aggregate is still a reference
                    motion / a proximity in [0, 1]; only *which* neighbours
                    count is learned.

    All modes use the same unfold path, so switching mode does not change the
    border handling (weights re-normalise on the valid taps at the border).
    """

    def __init__(self, size: int = 5, sigma: float = 1.0, mode: str = "adaptive", d_feat: int | None = None):
        super().__init__()
        if mode not in ("gaussian", "learned", "adaptive"):
            raise ValueError(f"unknown spatial kernel mode {mode!r}")
        if mode == "adaptive" and not d_feat:
            raise ValueError("adaptive spatial kernel needs the feature width")
        self.mode, self.size, self.pad = mode, size, size // 2
        base = gaussian_kernel(size, sigma).flatten()
        log_base = torch.log(base + 1e-12)
        if mode == "gaussian":
            self.register_buffer("log_kernel", log_base, persistent=False)
        elif mode == "learned":
            self.log_kernel = nn.Parameter(log_base.clone())
        else:
            self.predict = nn.Linear(d_feat, size * size)
            nn.init.zeros_(self.predict.weight)
            with torch.no_grad():
                self.predict.bias.copy_(log_base)

    def _taps(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Unfold ``[B, C, h, w]`` -> taps ``[B, C, K, h, w]`` and validity ``[B, 1, K, h, w]``."""
        b, c, h, w = x.shape
        k = self.size * self.size
        taps = F.unfold(x, self.size, padding=self.pad).view(b, c, k, h, w)
        ones = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        valid = F.unfold(ones, self.size, padding=self.pad).view(1, 1, k, h, w)
        return taps, valid

    def weights(self, x: Tensor, features: Tensor | None) -> Tensor:
        """Normalised tap weights ``[B, 1, K, h, w]`` (or ``[1, 1, K, 1, 1]`` when static)."""
        b, _, h, w = x.shape
        if self.mode == "adaptive":
            if features is None:
                raise ValueError("adaptive spatial kernel requires features [B, h, w, d]")
            logits = self.predict(features.to(self.predict.weight.dtype)).permute(0, 3, 1, 2)  # [B, K, h, w]
            logits = logits.unsqueeze(1).to(x.dtype)
        else:
            logits = self.log_kernel.to(x.dtype).view(1, 1, -1, 1, 1)
        return logits

    def forward(self, x: Tensor, features: Tensor | None = None) -> Tensor:
        """``x: [B, C, h, w]`` (``features: [B, h, w, d]`` for adaptive) -> same shape as ``x``."""
        taps, valid = self._taps(x)
        logits = self.weights(x, features).masked_fill(valid <= 0, float("-inf"))
        weights = logits.float().softmax(dim=2).to(x.dtype)
        return (taps * weights).sum(dim=2)


@dataclass
class CouplingField:
    displacement: Tensor       # [B, N, h, w, 3]
    body_reference: Tensor     # [B, N, h, w, 3]
    coupling_vector: Tensor    # [B, N, h, w, 3]  r
    coupling: Tensor           # [B, N, h, w]     c in [0, 1]
    transition: Tensor         # [B, N, h, w]     dc
    proximity: Tensor          # [B, N, h, w]     pi
    role: Tensor               # [B, N, h, w, 3]  rho (background, body, object)


def compute_coupling_field(displacement: Tensor, role: Tensor, kernel: SpatialKernel,
                           eps: float = 1e-6, features: Tensor | None = None) -> CouplingField:
    """Geometry of body vs. object motion; the only learnable part is the neighbourhood shape."""
    b, n, h, w, _ = displacement.shape
    rho_body = role[..., 1]
    flat = lambda x: x.reshape(b * n, h, w, -1).permute(0, 3, 1, 2)  # [B*N, C, h, w]
    feats = features.reshape(b * n, h, w, -1) if features is not None else None
    weighted = kernel(flat(displacement * rho_body[..., None]), feats)
    mass = kernel(flat(rho_body[..., None]), feats)
    body_ref = (weighted / (mass + eps)).permute(0, 2, 3, 1).reshape(b, n, h, w, 3)
    proximity = mass.permute(0, 2, 3, 1).reshape(b, n, h, w)
    r = displacement - body_ref
    c = r.norm(dim=-1) / (displacement.norm(dim=-1) + body_ref.norm(dim=-1) + eps)
    dc = torch.cat((torch.zeros_like(c[:, :1]), c[:, 1:] - c[:, :-1]), dim=1)
    return CouplingField(displacement, body_ref, r, c, dc, proximity, role)


PHYSICS_DIM = 6  # coupling, |transition|, proximity, |body ref|, camera |t|, camera angle


class SaliencyMLP(nn.Module):
    """``phi``: FiLM-conditioned MLP on the physical scalars -> unbounded log-saliency.

    The two camera scalars (ego-motion magnitude of the transition) let ``phi`` discount
    apparent motion when the head is moving; they are zero for static cameras / no camera tokens.
    Only differences of ``phi`` across the grid / the frames matter (see :class:`InteractionSaliency`).
    """

    def __init__(self, d_cond: int, hidden: int = 64):
        super().__init__()
        self.inp = nn.Linear(PHYSICS_DIM, hidden)
        self.film = nn.Linear(d_cond, 2 * hidden)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, physics: Tensor, cond: Tensor) -> Tensor:
        """``physics: [B, N, h, w, PHYSICS_DIM]``, ``cond: [B, d_cond]`` -> ``[B, N, h, w]``."""
        x = self.inp(physics)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        x = x * (1 + scale[:, None, None, None]) + shift[:, None, None, None]
        return self.out(x).squeeze(-1)


@dataclass
class Saliency:
    object_side: Tensor  # [B, N, h, w]  distribution over the grid per frame
    body_side: Tensor    # [B, N, h, w]  distribution over the grid per frame
    frame: Tensor        # [B, N]        relative transition strength (shift-invariant in phi)


class InteractionSaliency(nn.Module):
    def __init__(self, d_cond: int, kernel: SpatialKernel, hidden: int = 64, eps: float = 1e-6):
        super().__init__()
        self.phi = SaliencyMLP(d_cond, hidden)
        self.kernel = kernel
        self.eps = eps

    def forward(self, field: CouplingField, cond: Tensor, features: Tensor | None = None,
                camera_motion: Tensor | None = None) -> Saliency:
        """``camera_motion``: optional ``[B, N, 2]`` (translation norm, rotation angle) per transition."""
        b, n, h, w = field.coupling.shape
        if camera_motion is None:
            cam = torch.zeros(b, n, h, w, 2, dtype=field.coupling.dtype, device=field.coupling.device)
        else:
            if camera_motion.shape[:2] != (b, n):
                raise ValueError(f"camera_motion must be [B, {n}, 2], got {tuple(camera_motion.shape)}")
            cam = camera_motion.to(field.coupling.dtype)[:, :, None, None, :].expand(b, n, h, w, 2)
        physics = torch.cat((torch.stack((
            field.coupling, field.transition.abs(), field.proximity,
            field.body_reference.norm(dim=-1),
        ), dim=-1), cam), dim=-1)
        rho_object = field.role[..., 2]
        phi = self.phi(physics, cond)                                                    # [B, N, h, w]
        logits = torch.log(rho_object + self.eps) + phi
        s_object = (logits - torch.logsumexp(logits.flatten(2), dim=2)[:, :, None, None]).exp()
        feats = features.reshape(b * n, h, w, -1) if features is not None else None
        propagated = self.kernel((rho_object * s_object).reshape(b * n, 1, h, w), feats).reshape(b, n, h, w)
        s_body = propagated / (propagated.sum(dim=(2, 3), keepdim=True) + self.eps)
        frame = (rho_object * (phi - phi.flatten(1).max(dim=1).values[:, None, None, None]).exp()).sum(dim=(2, 3))
        return Saliency(s_object, s_body, frame)


__all__ = [
    "CouplingField", "InteractionSaliency", "Saliency", "SaliencyMLP", "SpatialKernel",
    "compute_coupling_field", "gaussian_kernel",
]
