"""Weight initialisation: interpolate a Wan-style expert from a wider / deeper one.

Used to warm-start the Track expert from the (Alpha) Video expert: blocks are
mapped by depth (``round(i * (L_src - 1) / (L_tgt - 1))``) and every tensor is
resized by linear interpolation along the mismatching axes.  The attention
space (``heads * head_dim``) is shared, so Q/K/V output rows are copied as-is
and only the residual-width axis is interpolated; the interpolated Track
therefore starts in the same attention space as the pretrained Video expert.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from metiswam4d.experts.local import MetisExpertCore, TrackExpert


class TensorSource(Protocol):
    def keys(self) -> Iterable[str]: ...
    def get_tensor(self, key: str) -> Tensor: ...


class DictSource:
    def __init__(self, state: Mapping[str, Tensor]):
        self._state = state

    def keys(self) -> Iterable[str]:
        return self._state.keys()

    def get_tensor(self, key: str) -> Tensor:
        return self._state[key]


def resize_tensor(source: Tensor, shape: torch.Size | tuple[int, ...]) -> Tensor:
    """Deterministically interpolate ``source`` to ``shape`` along mismatching axes."""
    shape = tuple(int(n) for n in shape)
    if tuple(source.shape) == shape:
        return source
    if source.ndim != len(shape):
        raise ValueError(f"rank mismatch {tuple(source.shape)} -> {shape}")
    x = source.float()
    for axis, (have, want) in enumerate(zip(x.shape, shape)):
        if have == want:
            continue
        moved = x.movedim(axis, -1)
        flat = moved.reshape(-1, 1, have)
        if have == 1:
            flat = flat.expand(-1, 1, want)
        else:
            flat = F.interpolate(flat, size=want, mode="linear", align_corners=True)
        x = flat.reshape(*moved.shape[:-1], want).movedim(-1, axis)
    return x.to(source.dtype)


def _map_layer(index: int, target_layers: int, source_layers: int) -> int:
    if target_layers == 1:
        return source_layers - 1
    return round(index * (source_layers - 1) / (target_layers - 1))


def source_key_for(target_key: str, target_layers: int, source_layers: int,
                   prefix: str = "") -> str | None:
    """Wan-compatible source key for one target key, or None if unmatched."""
    if target_key.startswith("blocks."):
        parts = target_key.split(".")
        parts[1] = str(_map_layer(int(parts[1]), target_layers, source_layers))
        return prefix + ".".join(parts)
    shared = ("text_embedding.", "time_embedding.", "time_projection.", "patch_embedding.", "head.")
    if target_key.startswith(shared):
        return prefix + target_key
    return None


@torch.no_grad()
def interpolate_from_source(
    target: MetisExpertCore,
    source: TensorSource | Mapping[str, Tensor],
    *,
    source_prefix: str = "",
    source_layers: int | None = None,
) -> dict[str, object]:
    """Fill ``target`` from Wan-style ``source`` tensors by mapping + resizing.

    ``source_prefix`` e.g. ``"video_backbone.dit."`` for an OpenWAM-Alpha
    safetensors file.  Returns a report with loaded / skipped / fresh keys.
    """
    if isinstance(source, Mapping):
        source = DictSource(source)
    keys = set(source.keys())
    if source_layers is None:
        block_ids = {int(k[len(source_prefix) + len("blocks."):].split(".")[0])
                     for k in keys if k.startswith(source_prefix + "blocks.")}
        if not block_ids:
            raise ValueError("source has no Wan blocks under the given prefix")
        source_layers = max(block_ids) + 1
    state = target.state_dict()
    loaded, resized, skipped, fresh = [], [], {}, []
    for name, param in state.items():
        src = source_key_for(name, target.num_layers, source_layers, source_prefix)
        if src is None or src not in keys:
            fresh.append(name)
            continue
        value = source.get_tensor(src)
        original_shape = tuple(value.shape)
        try:
            value = resize_tensor(value, param.shape)
        except ValueError as exc:
            skipped[name] = str(exc)
            continue
        param.copy_(value.to(dtype=param.dtype, device=param.device))
        (resized if tuple(param.shape) != original_shape else loaded).append(name)

    if isinstance(target, TrackExpert):
        # Condition stem: tile the (resized) patch embedding over the fused modalities.
        patch_w = state["patch_embedding.weight"]
        n = target.track_config.condition_modalities
        branch = patch_w / (n ** 0.5)
        state["condition_fusion.weight"].copy_(torch.cat([branch] * n, dim=1).to(
            state["condition_fusion.weight"]))
        state["condition_fusion.bias"].copy_(state["patch_embedding.bias"])
        fresh = [k for k in fresh if not k.startswith("condition_fusion.")]
        loaded.extend(["condition_fusion.weight", "condition_fusion.bias"])
    return {"loaded": loaded, "resized": resized, "skipped": skipped, "fresh": fresh,
            "source_layers": source_layers}


__all__ = ["DictSource", "TensorSource", "interpolate_from_source", "resize_tensor", "source_key_for"]
