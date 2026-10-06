"""Load OpenWAM-Alpha checkpoints into the local Wan-compatible experts.

An Alpha ``checkpoint_step_*.safetensors`` holds

    video_backbone.dit.*        Wan2.2-TI2V-5B DiT   -> LatentExpert (identical keys)
    action_backbone.*           separate Action DiT  -> ActionExpert (three renames)
    proprio_encoder.*           Linear(80 -> 4096)   -> MetisWAM4D.proprio_encoder
    video_backbone.vae.*, video_backbone.text_encoder.*   frozen, offline only

Loading streams tensor by tensor from the file so no second copy of the
~25 GB checkpoint is materialised.
"""
from __future__ import annotations

from pathlib import Path
import re

import torch
from torch import Tensor, nn

from metiswam4d.experts.local import ActionExpert, LatentExpert

VIDEO_PREFIX = "video_backbone.dit."
ACTION_PREFIX = "action_backbone."
PROPRIO_PREFIX = "proprio_encoder."

_ACTION_RENAMES = (
    (re.compile(r"^blocks\.(\d+)\.context_attn_norm\."), r"blocks.\1.norm3."),
    (re.compile(r"^time_embedding\.mlp\."), "time_embedding."),
    (re.compile(r"^time_projection\.proj\."), "time_projection."),
)


def action_key_from_alpha(key: str) -> str:
    """Map an ``action_backbone.*`` suffix to the local ``ActionExpert`` name."""
    for pattern, replacement in _ACTION_RENAMES:
        key = pattern.sub(replacement, key)
    return key


def alpha_key_from_action(key: str) -> str:
    key = re.sub(r"^blocks\.(\d+)\.norm3\.", r"blocks.\1.context_attn_norm.", key)
    key = re.sub(r"^time_embedding\.", "time_embedding.mlp.", key)
    key = re.sub(r"^time_projection\.", "time_projection.proj.", key)
    return key


def find_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.glob("checkpoint_step_*.safetensors"),
                        key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_step_*.safetensors under {path}")
    return candidates[-1]


@torch.no_grad()
def _stream_load(module: nn.Module, handle, prefix: str, rename, *, strict: bool) -> dict:
    target = module.state_dict()
    loaded, unexpected, mismatched = [], [], {}
    for key in handle.keys():
        if not key.startswith(prefix):
            continue
        name = rename(key[len(prefix):])
        if name not in target:
            unexpected.append(key)
            continue
        tensor = handle.get_tensor(key)
        if tuple(tensor.shape) != tuple(target[name].shape):
            mismatched[key] = (tuple(tensor.shape), tuple(target[name].shape))
            continue
        target[name].copy_(tensor.to(dtype=target[name].dtype, device=target[name].device))
        loaded.append(name)
    missing = sorted(set(target) - set(loaded))
    report = {"loaded": len(loaded), "missing": missing, "unexpected": unexpected,
              "mismatched": mismatched}
    if strict and (missing or mismatched):
        raise RuntimeError(f"strict Alpha load failed for prefix {prefix!r}: {report}")
    return report


@torch.no_grad()
def load_alpha_video(expert: LatentExpert, checkpoint: str | Path, *, strict: bool = True) -> dict:
    from safetensors import safe_open
    with safe_open(str(find_checkpoint(checkpoint)), framework="pt", device="cpu") as handle:
        return _stream_load(expert, handle, VIDEO_PREFIX, lambda k: k, strict=strict)


@torch.no_grad()
def load_alpha_action(expert: ActionExpert, checkpoint: str | Path, *, strict: bool = True) -> dict:
    from safetensors import safe_open
    with safe_open(str(find_checkpoint(checkpoint)), framework="pt", device="cpu") as handle:
        return _stream_load(expert, handle, ACTION_PREFIX, action_key_from_alpha, strict=strict)


@torch.no_grad()
def load_alpha_proprio(encoder: nn.Linear, checkpoint: str | Path) -> dict:
    from safetensors import safe_open
    with safe_open(str(find_checkpoint(checkpoint)), framework="pt", device="cpu") as handle:
        return _stream_load(encoder, handle, PROPRIO_PREFIX, lambda k: k, strict=True)


class AlphaVideoSource:
    """``TensorSource`` view of the Alpha video DiT for interpolation init."""

    def __init__(self, checkpoint: str | Path):
        from safetensors import safe_open
        self._handle = safe_open(str(find_checkpoint(checkpoint)), framework="pt", device="cpu")
        self._keys = [k for k in self._handle.keys() if k.startswith(VIDEO_PREFIX)]

    def keys(self):
        return list(self._keys)

    def get_tensor(self, key: str) -> Tensor:
        return self._handle.get_tensor(key)

    def close(self) -> None:
        self._handle.__exit__(None, None, None)


__all__ = [
    "ACTION_PREFIX", "PROPRIO_PREFIX", "VIDEO_PREFIX", "AlphaVideoSource",
    "action_key_from_alpha", "alpha_key_from_action", "find_checkpoint",
    "load_alpha_action", "load_alpha_proprio", "load_alpha_video",
]
