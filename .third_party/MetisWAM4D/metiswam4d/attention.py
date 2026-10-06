"""Three-expert token layout and visibility rules for one joint softmax.

Key order inside a macro layer::

    [video_clean | video_future | track_clean | track_future | action]

``video_clean`` is the clean first latent frame, ``track_clean`` the RGB /
depth / mask condition tokens plus the zero-displacement anchor frame.

Rules (True = may attend):

* clean conditions read only themselves: they must not change with the noisy
  future they condition;
* every future world token reads all world tokens and the Action tokens, so a
  clean Action block conditions Video / Track prediction;
* Action always reads itself and the clean current-observation tokens; its
  reading of the *future* world inside the joint softmax is ``dense`` (all
  future Video / Track tokens, the OpenWAM-Alpha behaviour) or absent
  (``compact`` / ``none``).  In ``compact`` mode the K compact interaction
  tokens reach Action through a separate zero-initialised gated cross-attention
  (see ``model.FocusCrossAttention``), so at initialisation Action is exactly
  blind to the future world;
* ``dense_bias`` (compact mode only) additionally exposes the dense future keys
  to Action with an additive logit bias, used to anneal dense -> compact
  (bias 0 = dense, bias -> -inf = compact).  The mask is then a float additive
  mask instead of a boolean one;
* modality dropout hides a dropped future branch's keys from every other
  expert while the dropped branch keeps reading and keeps its own loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

KEY_ORDER: tuple[str, ...] = (
    "video_clean", "video_future", "track_clean", "track_future", "action",
)
QUERY_SEGMENTS: tuple[str, ...] = (
    "video_clean", "video_future", "track_clean", "track_future", "action",
)
ACTION_READ_MODES: tuple[str, ...] = ("dense", "compact", "none")


@dataclass(frozen=True)
class TokenLayout:
    """Ordered named segments of one token sequence."""

    segments: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.segments]
        if len(set(names)) != len(names):
            raise ValueError("duplicate segment names")
        if any(length < 0 for _, length in self.segments):
            raise ValueError("segment lengths must be non-negative")

    @classmethod
    def from_lengths(cls, order: tuple[str, ...], lengths: dict[str, int]) -> "TokenLayout":
        return cls(tuple((name, int(lengths.get(name, 0))) for name in order))

    @property
    def total(self) -> int:
        return sum(length for _, length in self.segments)

    def has(self, name: str) -> bool:
        return self.length(name) > 0

    def length(self, name: str) -> int:
        for segment, length in self.segments:
            if segment == name:
                return length
        return 0

    def slice(self, name: str) -> slice:
        start = 0
        for segment, length in self.segments:
            if segment == name:
                return slice(start, start + length)
            start += length
        raise KeyError(name)

    def split(self, tokens: Tensor, dim: int = 1) -> dict[str, Tensor]:
        pieces = tokens.split([length for _, length in self.segments], dim=dim)
        return {name: piece for (name, _), piece in zip(self.segments, pieces)}


def build_visibility(
    keys: TokenLayout,
    queries: TokenLayout,
    *,
    batch_size: int,
    action_read: str = "dense",
    drop_video: Tensor | None = None,
    drop_track: Tensor | None = None,
    clean_video: Tensor | None = None,
    clean_track: Tensor | None = None,
    clean_action: Tensor | None = None,
    dense_bias: float | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """``[B, 1, Q, K]`` visibility mask for the joint softmax.

    Boolean unless ``dense_bias`` is given (compact mode annealing), in which
    case a float additive mask is returned: 0 for visible pairs, ``dense_bias``
    for Action -> dense-future pairs, -inf elsewhere.

    ``drop_*`` hide a future world branch from the other experts (modality
    dropout).  ``clean_*`` mark samples whose block of that modality is at
    sigma = 0: it is then a pure condition and does not read any noisy future
    (it still reads itself and the clean current observation).
    """
    if action_read not in ACTION_READ_MODES:
        raise ValueError(f"action_read must be one of {ACTION_READ_MODES}")
    if dense_bias is not None and action_read != "compact":
        raise ValueError("dense_bias is only meaningful in compact mode")
    if keys.total < 1 or queries.total < 1:
        raise ValueError("keys and queries must contain tokens")

    def per_sample(flag: Tensor | None) -> Tensor:
        if flag is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if flag.shape != (batch_size,) or flag.dtype != torch.bool:
            raise ValueError("per-sample flags must be bool tensors with one entry per sample")
        return flag.to(device)

    keep_video = ~per_sample(drop_video)
    keep_track = ~per_sample(drop_track)
    open_video = ~per_sample(clean_video)   # noisy video reads the world
    open_track = ~per_sample(clean_track)
    open_action = ~per_sample(clean_action)

    mask = torch.zeros(batch_size, 1, queries.total, keys.total, dtype=torch.bool, device=device)

    def allow(query: str, key: str, flag: Tensor | None = None) -> None:
        if not (queries.has(query) and keys.has(key)):
            return
        qs, ks = queries.slice(query), keys.slice(key)
        if flag is None:
            mask[:, :, qs, ks] = True
        else:
            mask[:, :, qs, ks] = flag[:, None, None, None]

    # Clean current-observation conditions are self-contained.
    allow("video_clean", "video_clean")
    allow("track_clean", "track_clean")

    # Future Video: own tokens always; the rest only while noisy, subject to dropout.
    allow("video_future", "video_clean")
    allow("video_future", "video_future")
    allow("video_future", "track_clean", open_video)
    allow("video_future", "track_future", open_video & keep_track)
    allow("video_future", "action", open_video)

    # Future Track: symmetric.
    allow("track_future", "track_clean")
    allow("track_future", "track_future")
    allow("track_future", "video_clean", open_track)
    allow("track_future", "video_future", open_track & keep_video)
    allow("track_future", "action", open_track)

    # Action: itself and the clean current observation always; the dense future
    # world only in dense mode and only while the action block is noisy.
    allow("action", "action")
    allow("action", "video_clean")
    allow("action", "track_clean")
    if action_read == "dense":
        allow("action", "video_future", open_action & keep_video)
        allow("action", "track_future", open_action & keep_track)

    if not bool(mask.any(dim=-1).all()):
        raise RuntimeError("every query must see at least one key")
    if dense_bias is None:
        return mask

    # Annealing: expose the dense future keys to Action with an additive bias.
    extra = torch.zeros_like(mask)
    if queries.has("action"):
        qs = queries.slice("action")
        for key, keep in (("video_future", keep_video), ("track_future", keep_track)):
            if keys.has(key):
                extra[:, :, qs, keys.slice(key)] = (open_action & keep)[:, None, None, None]
    extra &= ~mask
    additive = torch.full(mask.shape, float("-inf"), device=mask.device)
    additive[mask] = 0.0
    additive[extra] = float(dense_bias)
    return additive


def dense_to_compact_bias(step: int, anneal_steps: int, max_bias: float = 12.0) -> float | None:
    """Additive logit bias on the dense future keys during dense -> compact annealing.

    Linear from 0 (pure dense reading) to ``-max_bias`` over ``anneal_steps``; ``None`` afterwards
    (pure compact).  ``exp(-12) ~ 6e-6``: the dense keys are effectively gone at the end of the ramp.
    """
    if anneal_steps <= 0 or step >= anneal_steps:
        return None
    return -max_bias * step / anneal_steps


def joint_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    *,
    heads: int,
) -> Tensor:
    """Masked scaled-dot-product attention over ``[B, S, heads * head_dim]`` streams."""
    batch, q_len, width = query.shape
    if key.shape != value.shape or key.shape[0] != batch or key.shape[2] != width:
        raise ValueError("all expert Q/K/V must share batch and attention width")
    if width % heads:
        raise ValueError("attention width must be divisible by the head count")
    if mask.shape[-2:] != (q_len, key.shape[1]):
        raise ValueError("visibility mask does not match query/key lengths")

    def split(x: Tensor) -> Tensor:
        return x.reshape(batch, x.shape[1], heads, width // heads).transpose(1, 2)

    if mask.dtype != torch.bool:
        mask = mask.to(query.dtype)
    out = F.scaled_dot_product_attention(split(query), split(key), split(value), attn_mask=mask)
    return out.transpose(1, 2).reshape(batch, q_len, width)


__all__ = [
    "ACTION_READ_MODES",
    "KEY_ORDER",
    "QUERY_SEGMENTS",
    "TokenLayout",
    "build_visibility",
    "dense_to_compact_bias",
    "joint_attention",
]
