"""GPU-side conversion of raw RT2 batches into the model contract (Wan2.2 VAE latents).

Video: the 9-frame 384x320 layout -> ``video_clean [B, 48, 3, 24, 20]``.
Track: 9-frame mu-law RGB (anchor + 8 transitions) centre-padded 240 -> 256 rows -> ``track_clean
[B, 48, 3, 16, 20]``; RGB / depth / mask condition images of the current head frame -> ``[B, 48, 1, 16, 20]``.
Latent-grid supervision: ``track_valid`` (any foreground in the frame group, max-pooled), ``track_role``
(majority role), and the frame-level token-grid targets ``track_disp_frames [B, 8, 8, 10, 3]`` /
``track_role_frames [B, 8, 8, 10]`` for the reading interface.

Latents follow Alpha: ``(mode - latents_mean) / latents_std``; the causal Wan VAE encodes the first
frame independently of the rest, so anchor / first-frame latents equal the whole-clip encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from metiswam4d.data.camera import identity_code


@dataclass
class RT2EncoderConfig:
    vae_path: str = "/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    depth_stats: str = "/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D/depth_stats.json"
    track_height: int = 256   # 240 rows centre-padded to a multiple of 16
    track_width: int = 320
    token_patch: int = 2      # latent 16x20 -> tokens 8x10
    dtype: str = "bf16"
    unknown_role: bool = False  # role 0 = unknown (RoboDojo): token cells without foreground get role target -1 (ignored)


def center_pad(x: Tensor, height: int, width: int, fill: float = 0.0) -> Tensor:
    """Pad ``[..., H, W, C]`` to ``height x width`` without resampling."""
    h, w = x.shape[-3], x.shape[-2]
    top, left = (height - h) // 2, (width - w) // 2
    out = x.new_full((*x.shape[:-3], height, width, x.shape[-1]), fill)
    out[..., top:top + h, left:left + w, :] = x
    return out


def majority_pool(labels: Tensor, out_hw: tuple[int, int], classes: int = 3) -> Tensor:
    """``[B, T, H, W]`` int labels -> majority class per output cell ``[B, T, h, w]``."""
    b, t, h, w = labels.shape
    onehot = F.one_hot(labels.long().clamp(0, classes - 1), classes).permute(0, 1, 4, 2, 3).float()
    pooled = F.adaptive_avg_pool2d(onehot.reshape(b * t, classes, h, w), out_hw)
    return pooled.argmax(dim=1).reshape(b, t, *out_hw).to(torch.int8)


class RT2OnlineEncoder:
    def __init__(self, config: RT2EncoderConfig, device: torch.device):
        from diffusers import AutoencoderKLWan
        self.config = config
        self.device = device
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config.dtype]
        self.vae = AutoencoderKLWan.from_pretrained(str(Path(config.vae_path) / "vae"), torch_dtype=self.dtype)
        self.vae = self.vae.eval().requires_grad_(False).to(device)
        self.mean = torch.as_tensor(self.vae.config.latents_mean, device=device, dtype=self.dtype).view(1, -1, 1, 1, 1)
        self.std = torch.as_tensor(self.vae.config.latents_std, device=device, dtype=self.dtype).view(1, -1, 1, 1, 1)
        stats = json.loads(Path(config.depth_stats).read_text())["head_camera"]
        self.depth_min, self.depth_max = float(stats["min_m"]), float(stats["max_m"])

    @torch.no_grad()
    def encode_pixels(self, pixels_u8: Tensor) -> Tensor:
        """uint8 ``[B, T, H, W, 3]`` -> normalised latents ``[B, 48, T', H/16, W/16]``."""
        x = pixels_u8.to(self.device).permute(0, 4, 1, 2, 3).to(self.dtype).mul(2.0 / 255.0).sub(1.0)
        z = self.vae.encode(x).latent_dist.mode()
        return ((z - self.mean) / self.std).to(self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: Tensor) -> Tensor:
        """Normalised latents ``[B, 48, T', h, w]`` -> uint8 pixels ``[B, T, H, W, 3]`` (inverse of ``encode_pixels``)."""
        z = latents.to(device=self.device, dtype=self.dtype) * self.std + self.mean
        x = self.vae.decode(z).sample.float().clamp(-1.0, 1.0)
        return x.add(1.0).mul(127.5).round().to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()

    def condition_pixels(self, head_rgb: Tensor, depth_mm: Tensor, mask: Tensor) -> Tensor:
        """Stack RGB / depth / mask condition images as ``[3B, 1, 256, 320, 3]`` uint8."""
        b = head_rgb.shape[0]
        depth_m = depth_mm.float() / 1000.0
        depth_u8 = ((depth_m - self.depth_min) / (self.depth_max - self.depth_min)).clamp(0, 1).mul(255).round().to(torch.uint8)
        depth_rgb = depth_u8[..., None].expand(*depth_u8.shape, 3)
        mask_rgb = mask.to(torch.uint8).mul(255)[..., None].expand(*mask.shape, 3)
        stacked = torch.cat((head_rgb.to(torch.uint8), depth_rgb, mask_rgb), dim=0)[:, None]  # [3B, 1, H, W, 3]
        return center_pad(stacked, self.config.track_height, self.config.track_width)

    @torch.no_grad()
    def __call__(self, raw: dict) -> dict:
        b = raw["video_frames"].shape[0]
        out: dict = {}
        out["video_clean"] = self.encode_pixels(raw["video_frames"])
        track_px = center_pad(raw["track_rgb"], self.config.track_height, self.config.track_width)
        out["track_clean"] = self.encode_pixels(track_px)
        cond = self.encode_pixels(self.condition_pixels(raw["head_rgb"], raw["head_depth_mm"], raw["head_mask"]))
        out["rgb_condition"], out["depth_condition"], out["mask_condition"] = cond[:b], cond[b:2 * b], cond[2 * b:]

        lat_h, lat_w = out["track_clean"].shape[-2:]
        fg = center_pad(raw["track_foreground"][..., None].to(torch.uint8), self.config.track_height,
                        self.config.track_width)[..., 0].bool().to(self.device)                 # [B, 9, 256, 320]
        roles = center_pad(raw["track_role_px"][..., None], self.config.track_height,
                           self.config.track_width)[..., 0].to(self.device)                    # [B, 9, 256, 320]
        # Latent temporal groups of the causal Wan VAE: anchor | transitions 1:5 | 5:9.
        groups = [(0, 1), (1, 5), (5, 9)]
        valid_groups = torch.stack([fg[:, a:z].any(dim=1) for a, z in groups], dim=1).float()  # [B, 3, H, W]
        out["track_valid"] = F.adaptive_max_pool2d(valid_groups, (lat_h, lat_w)).bool()[:, None]  # [B, 1, 3, h, w]
        role_groups = torch.stack([roles[:, a:z].max(dim=1).values for a, z in groups], dim=1)  # [B, 3, H, W]
        out["track_role"] = majority_pool(role_groups, (lat_h, lat_w))                             # [B, 3, h, w]
        # A mixed RoboDojo batch can contain object-labelled demonstrations and
        # robot-only self-play. Unlabelled rollout pixels are not known static background.
        unknown = torch.full((b,), self.config.unknown_role, dtype=torch.bool, device=self.device)
        if raw.get("track_unknown_role") is not None:
            unknown = unknown | raw["track_unknown_role"].to(device=self.device, dtype=torch.bool)
        out["track_role"] = torch.where(unknown[:, None, None, None] & (out["track_role"] == 0),
                                         torch.full_like(out["track_role"], -1), out["track_role"])

        tok_h, tok_w = lat_h // self.config.token_patch, lat_w // self.config.token_patch
        delta = center_pad(raw["track_delta"].float(), self.config.track_height, self.config.track_width).to(self.device)  # [B, 8, 256, 320, 3]
        fg8 = fg[:, 1:].float()                                                                     # transitions
        num = F.adaptive_avg_pool2d((delta * fg8[..., None]).permute(0, 1, 4, 2, 3).flatten(0, 1), (tok_h, tok_w))
        den = F.adaptive_avg_pool2d(fg8.flatten(0, 1)[:, None], (tok_h, tok_w)).clamp(min=1e-6)
        disp = (num / den).reshape(b, 8, 3, tok_h, tok_w).permute(0, 1, 3, 4, 2)                 # [B, 8, h, w, 3]
        cell_fg = F.adaptive_max_pool2d(fg8.flatten(0, 1)[:, None], (tok_h, tok_w)).reshape(b, 8, tok_h, tok_w) > 0
        out["track_disp_frames"] = torch.where(cell_fg[..., None], disp, torch.zeros_like(disp))
        out["track_role_frames"] = majority_pool(roles[:, 1:], (tok_h, tok_w))                    # [B, 8, h, w]
        fill = torch.where(unknown, -1, 0).to(out["track_role_frames"].dtype)[:, None, None, None]
        out["track_role_frames"] = torch.where(cell_fg, out["track_role_frames"], fill)
        # RoboTwin head camera is rigidly mounted: identity ego-motion for every transition.
        out["camera_delta"] = identity_code(b, 8, device=self.device)
        out["camera_valid"] = torch.ones(b, 8, dtype=torch.bool, device=self.device)

        for key in ("text_context", "text_mask", "action", "action_mask", "proprio", "proprio_mask", "embodiment",
                    "progress_video", "progress_body"):
            if raw.get(key) is not None:
                out[key] = raw[key].to(self.device)
        out["key"] = raw.get("key")
        return out


__all__ = ["RT2EncoderConfig", "RT2OnlineEncoder", "center_pad", "majority_pool"]
