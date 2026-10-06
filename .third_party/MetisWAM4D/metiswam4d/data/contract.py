"""Model-side sample contract.

Every training sample is a dict of per-sample tensors (no batch dimension).
All datasets (human ego pretraining, IG-10K, RoboTwin2, RoboDojo, real robot)
produce this same layout; the offline preprocessing (video -> Track4D -> VAE
latents, text -> UMT5 cache) lives outside this package.

Required for a Video block:   video_clean [C, F_v, H_v, W_v]
Required for a Track block:   track_clean [C, F_t, H_t, W_t], rgb_condition /
                              depth_condition / mask_condition [C, 1, H_t, W_t]
Optional Track supervision:   track_valid [1, F_t, H_t, W_t] bool,
                              track_role [F_t, H_t, W_t] int8 {0 bg, 1 body, 2 object},
                              track_disp_frames [N_f, h, w, 3] float (token grid, normalised),
                              track_role_frames [N_f, h, w] int8
Optional camera ego-motion:   camera_delta [N_f, 9] float (see data/camera.py; zeros = static),
                              camera_valid [N_f] bool (supervise the camera tokens)
Optional task progress:       progress_video [1] float in [0, 1] (current video frame / episode length),
                              progress_body [1] float (current embodiment state / its episode length)
Required for an Action block: action [H, A], action_mask [H, A] bool,
                              proprio [1, A], proprio_mask [1, A] bool
Always:                       text_context [L, text_dim], embodiment (int), key (str)

The window follows OpenWAM-Alpha: 33 source frames at stride 4 -> F = 1 + 8/kappa
latent frames (kappa = 4), N_f = 8 frame-level slots, H = 32 action steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SampleSpec:
    video_channels: int = 48
    video_frames: int = 3
    video_height: int = 24
    video_width: int = 20
    track_channels: int = 48
    track_frames: int = 3
    track_height: int = 16
    track_width: int = 20
    kappa: int = 4
    token_patch: tuple[int, int] = (2, 2)
    action_horizon: int = 32
    action_dim: int = 80
    proprio_dim: int = 80
    text_dim: int = 4096
    max_text_len: int = 512

    @property
    def frame_slots(self) -> int:
        return (self.track_frames - 1) * self.kappa

    @property
    def track_token_grid(self) -> tuple[int, int]:
        return self.track_height // self.token_patch[0], self.track_width // self.token_patch[1]

    def __post_init__(self) -> None:
        if self.video_frames != self.track_frames:
            raise ValueError("Video and Track must cover the same latent frame count")
        if self.action_horizon % self.frame_slots:
            raise ValueError("action horizon must be a multiple of the frame slots")


MODALITY_KEYS = {
    "video": ("video_clean",),
    "track": ("track_clean", "rgb_condition", "depth_condition", "mask_condition"),
    "action": ("action", "action_mask", "proprio", "proprio_mask"),
}
OPTIONAL_TRACK_KEYS = ("track_valid", "track_role", "track_disp_frames", "track_role_frames",
                       "camera_delta", "camera_valid")
CAMERA_DIM = 9


def presence(sample: dict[str, Any]) -> tuple[str, ...]:
    return tuple(m for m, keys in MODALITY_KEYS.items() if sample.get(keys[0]) is not None)


def _check(tensor: Tensor, shape: Sequence[int], name: str, dtype: torch.dtype | None = None) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {tensor.dtype}")


def validate_sample(sample: dict[str, Any], spec: SampleSpec) -> tuple[str, ...]:
    """Raise on contract violations; return the present modalities."""
    present = presence(sample)
    if not present:
        raise ValueError("sample has no generative modality")
    if sample.get("text_context") is None:
        raise ValueError("text_context is required")
    ctx = sample["text_context"]
    if ctx.ndim != 2 or ctx.shape[1] != spec.text_dim or ctx.shape[0] > spec.max_text_len:
        raise ValueError(f"text_context must be [L<={spec.max_text_len}, {spec.text_dim}], got {tuple(ctx.shape)}")
    if "video" in present:
        _check(sample["video_clean"], (spec.video_channels, spec.video_frames, spec.video_height, spec.video_width), "video_clean")
    if "track" in present:
        shape = (spec.track_channels, spec.track_frames, spec.track_height, spec.track_width)
        _check(sample["track_clean"], shape, "track_clean")
        for key in ("rgb_condition", "depth_condition", "mask_condition"):
            if sample.get(key) is None:
                raise ValueError(f"Track requires {key}")
            _check(sample[key], (spec.track_channels, 1, spec.track_height, spec.track_width), key)
        h, w = spec.track_token_grid
        if sample.get("track_valid") is not None:
            _check(sample["track_valid"], (1, spec.track_frames, spec.track_height, spec.track_width), "track_valid", torch.bool)
        if sample.get("track_role") is not None:
            _check(sample["track_role"], (spec.track_frames, spec.track_height, spec.track_width), "track_role")
        if sample.get("track_disp_frames") is not None:
            _check(sample["track_disp_frames"], (spec.frame_slots, h, w, 3), "track_disp_frames")
        if sample.get("track_role_frames") is not None:
            _check(sample["track_role_frames"], (spec.frame_slots, h, w), "track_role_frames")
        if sample.get("camera_delta") is not None:
            _check(sample["camera_delta"], (spec.frame_slots, CAMERA_DIM), "camera_delta")
            if sample.get("camera_valid") is None:
                raise ValueError("camera_delta requires camera_valid")
            _check(sample["camera_valid"], (spec.frame_slots,), "camera_valid", torch.bool)
    for key in ("progress_video", "progress_body"):
        if sample.get(key) is not None:
            _check(sample[key], (1,), key)
            if not (0.0 <= float(sample[key][0]) <= 1.0):
                raise ValueError(f"{key} must lie in [0, 1]")
    if "action" in present:
        _check(sample["action"], (spec.action_horizon, spec.action_dim), "action")
        _check(sample["action_mask"], (spec.action_horizon, spec.action_dim), "action_mask", torch.bool)
        _check(sample["proprio"], (1, spec.proprio_dim), "proprio")
        _check(sample["proprio_mask"], (1, spec.proprio_dim), "proprio_mask", torch.bool)
    return present


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a homogeneous list of samples; pad text to the longest prompt."""
    if not samples:
        raise ValueError("empty batch")
    signature = presence(samples[0])
    if any(presence(s) != signature for s in samples):
        raise ValueError("a batch must contain samples with identical modality presence")
    batch: dict[str, Any] = {}
    lengths = [s["text_context"].shape[0] for s in samples]
    l_max = max(lengths)
    ctx = samples[0]["text_context"]
    text = torch.zeros(len(samples), l_max, ctx.shape[1], dtype=ctx.dtype)
    mask = torch.zeros(len(samples), l_max, dtype=torch.bool)
    for i, s in enumerate(samples):
        text[i, :lengths[i]] = s["text_context"]
        mask[i, :lengths[i]] = True
    batch["text_context"], batch["text_mask"] = text, mask
    tensor_keys = [k for k in samples[0] if k not in ("text_context", "key") and torch.is_tensor(samples[0][k])]
    for key in tensor_keys:
        if any(s.get(key) is None for s in samples):
            continue  # optional supervision missing for some samples -> drop for the batch
        batch[key] = torch.stack([s[key] for s in samples])
    batch["key"] = [s.get("key", str(i)) for i, s in enumerate(samples)]
    if "embodiment" in samples[0]:
        batch["embodiment"] = torch.as_tensor([int(s["embodiment"]) for s in samples])
    return batch


def move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype | None = None) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if dtype is not None and value.is_floating_point() and key != "text_context":
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        out[key] = value
    return out


__all__ = ["CAMERA_DIM", "MODALITY_KEYS", "OPTIONAL_TRACK_KEYS", "SampleSpec", "collate", "move_batch", "presence", "validate_sample"]
