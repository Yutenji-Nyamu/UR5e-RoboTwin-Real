"""Coupling-Salient Interaction Attention (CSIA).

``K`` task/state-conditioned queries read, in every sub-frame ``n``, the
object-side and the body-side tokens of one world modality under a saliency
log-bias.  Saliency decides *which body-object pairs are worth reading*, the
content score decides *what to read within them*.  The two attention outputs
and the saliency-weighted coupling vector are fused into one interaction token
per (query, sub-frame).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CSIAOutput:
    tokens: Tensor            # [B, K, N, d]  e^m_{k,n}
    alpha_object: Tensor      # [B, K, N, P]  object-side attention (P = h*w)
    alpha_body: Tensor        # [B, K, N, P]


class CSIA(nn.Module):
    def __init__(self, d: int, num_queries: int, d_cond: int, gamma_init: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.K = num_queries
        self.eps = eps
        self.queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)
        self.query_cond = nn.Linear(d_cond, d)
        nn.init.zeros_(self.query_cond.weight)
        nn.init.zeros_(self.query_cond.bias)
        self.role_embedding = nn.Embedding(2, d)  # 0 object, 1 body
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        # gamma >= 0 through softplus; inverse-softplus initialisation (1-element: FSDP2 rejects 0-d params).
        self.gamma_raw = nn.Parameter(torch.full((1,), math.log(math.expm1(gamma_init))))
        self.coupling_proj = nn.Linear(3, d)
        self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))

    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.gamma_raw)[0]

    def query_vectors(self, cond: Tensor) -> Tensor:
        """``[B, K, d]`` task/state-modulated queries (shared with TAA)."""
        return self.queries[None] + self.query_cond(cond)[:, None]

    def _attend(self, q: Tensor, keys: Tensor, values: Tensor, saliency: Tensor | None,
                use_saliency: Tensor | None):
        """q [B,K,d]; keys/values [B,N,P,d]; saliency [B,N,P] or None -> (out [B,K,N,d], alpha).

        ``use_saliency`` ``[B]`` bool switches the log-bias off per sample (Track
        dropped or absent: attention becomes content-only).
        """
        logits = torch.einsum("bkd,bnpd->bknp", q, keys) / math.sqrt(self.d)
        if saliency is not None:
            bias = self.gamma * torch.log(saliency + self.eps)[:, None]
            if use_saliency is not None:
                bias = bias * use_saliency.to(bias.dtype)[:, None, None, None]
            logits = logits + bias
        alpha = logits.softmax(dim=-1)
        return torch.einsum("bknp,bnpd->bknd", alpha, values), alpha

    def forward(
        self,
        features: Tensor,                 # [B, N, h, w, d] sub-frame features of one modality
        cond: Tensor,                     # [B, d_cond]
        saliency_object: Tensor | None,   # [B, N, h, w] (None: content only)
        saliency_body: Tensor | None,     # [B, N, h, w]
        coupling_vector: Tensor | None,   # [B, N, h, w, 3] on this grid, or None
        object_alpha: Tensor | None = None,  # externally supplied [B, K, N, P] (Video uses Track's)
        use_saliency: Tensor | None = None,  # [B] bool
    ) -> CSIAOutput:
        b, n, h, w, d = features.shape
        flat = features.reshape(b, n, h * w, d)
        keys, values = self.key(flat), self.value(flat)
        q = self.query_vectors(cond)
        s_obj = saliency_object.reshape(b, n, h * w) if saliency_object is not None else None
        s_body = saliency_body.reshape(b, n, h * w) if saliency_body is not None else None
        a_obj, alpha_obj = self._attend(q + self.role_embedding.weight[0], keys, values, s_obj, use_saliency)
        a_body, alpha_body = self._attend(q + self.role_embedding.weight[1], keys, values, s_body, use_saliency)
        if coupling_vector is not None:
            weights = object_alpha if object_alpha is not None else alpha_obj
            r_bar = torch.einsum("bknp,bnpc->bknc", weights, coupling_vector.reshape(b, n, h * w, 3))
            if use_saliency is not None:
                r_bar = r_bar * use_saliency.to(r_bar.dtype)[:, None, None, None]
        else:
            r_bar = torch.zeros(b, self.K, n, 3, dtype=features.dtype, device=features.device)
        tokens = self.fuse(torch.cat((a_obj, a_body, self.coupling_proj(r_bar)), dim=-1))
        return CSIAOutput(tokens, alpha_obj, alpha_body)


__all__ = ["CSIA", "CSIAOutput"]
