"""GPU-side conversion of raw human windows into the model contract (Wan2.2 VAE latents + UMT5 text).

Video: 9 frames 512x288 -> ``video_clean [B, 48, 3, 18, 32]``.  Track (when present): 9-frame mu-law RGB
-> ``track_clean [B, 48, 3, 18, 32]``; RGB / depth / mask condition images of the current frame ->
``[B, 48, 1, 18, 32]`` with ``condition_present [B, 3]`` (Kling has no depth: the Track expert substitutes
its learned missing-modality token and the reconstruction loss skips it).  Latent-grid supervision as in
the RT2 encoder (``track_valid``, ``track_role``, ``track_disp_frames``, ``track_role_frames``), camera
codes with ``camera_valid``, progress targets with ``progress_valid``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from metiswam4d.data.human.hand_render import hand_pixels
from metiswam4d.data.human.text import UMT5Online
from metiswam4d.data.rt2.online_encoder import majority_pool


@dataclass
class HumanEncoderConfig:
    wan_path: str = "/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    depth_min_m: float = 0.2      # linear depth-to-intensity range of the condition image (IG-10K tabletop: 0.4-1.1 m)
    depth_max_m: float = 1.6
    token_patch: int = 2          # latent 18x32 -> tokens 9x16
    max_text_len: int = 256
    dtype: str = "bf16"


class HumanOnlineEncoder:
    def __init__(self, config: HumanEncoderConfig, device: torch.device):
        from diffusers import AutoencoderKLWan
        self.config = config
        self.device = device
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config.dtype]
        self.vae = AutoencoderKLWan.from_pretrained(str(Path(config.wan_path) / "vae"), torch_dtype=self.dtype)
        self.vae = self.vae.eval().requires_grad_(False).to(device)
        self.mean = torch.as_tensor(self.vae.config.latents_mean, device=device, dtype=self.dtype).view(1, -1, 1, 1, 1)
        self.std = torch.as_tensor(self.vae.config.latents_std, device=device, dtype=self.dtype).view(1, -1, 1, 1, 1)
        self.text = UMT5Online(config.wan_path, device, max_len=config.max_text_len, dtype=self.dtype)

    @torch.no_grad()
    def encode_pixels(self, pixels_u8: Tensor) -> Tensor:
        """uint8 ``[B, T, H, W, 3]`` -> normalised latents ``[B, 48, T', H/16, W/16]``."""
        x = pixels_u8.to(self.device).permute(0, 4, 1, 2, 3).to(self.dtype).mul(2.0 / 255.0).sub(1.0)
        z = self.vae.encode(x).latent_dist.mode()
        return ((z - self.mean) / self.std).to(self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: Tensor) -> Tensor:
        z = latents.to(device=self.device, dtype=self.dtype) * self.std + self.mean
        x = self.vae.decode(z).sample.float().clamp(-1.0, 1.0)
        return x.add(1.0).mul(127.5).round().to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()

    def condition_pixels(self, head_rgb: Tensor, depth_m: Tensor, mask: Tensor) -> Tensor:
        """``[3B, 1, H, W, 3]`` uint8: RGB | depth intensity | mask."""
        c = self.config
        depth_u8 = ((depth_m.float() - c.depth_min_m) / (c.depth_max_m - c.depth_min_m)).clamp(0, 1).mul(255).round().to(torch.uint8)
        depth_rgb = depth_u8[..., None].expand(*depth_u8.shape, 3)
        mask_rgb = mask.to(torch.uint8).mul(255)[..., None].expand(*mask.shape, 3)
        return torch.cat((head_rgb.to(torch.uint8), depth_rgb, mask_rgb), dim=0)[:, None]

    def render_hands(self, raw: dict) -> dict:
        """Rasterise the Kling hand meshes of the batch on the GPU into the track pixel fields (samples without
        meshes, i.e. IG-10K in a mixed batch, keep their own pixels)."""
        height, width = raw["video_frames"].shape[2:4]
        pixels = hand_pixels(raw["hand_verts"], raw["hand_colors"], raw["hand_is_right"], raw["hand_present"],
                             height, width, device=self.device)
        from_mesh = raw["hand_present"].flatten(1).any(dim=1).to(self.device)
        raw = dict(raw)
        for key, value in pixels.items():
            if raw.get(key) is None:
                raw[key] = value
            else:
                raw[key] = torch.where(from_mesh.view(-1, *[1] * (value.dim() - 1)), value, raw[key].to(self.device))
        return raw

    @torch.no_grad()
    def __call__(self, raw: dict) -> dict:
        b = raw["video_frames"].shape[0]
        out: dict = {}
        out["video_clean"] = self.encode_pixels(raw["video_frames"])
        out["text_context"], out["text_mask"] = self.text(list(raw["prompt"]))
        for key in ("embodiment", "progress_video", "progress_body", "progress_valid",
                    "action", "action_mask", "proprio", "proprio_mask"):  # robot windows: already in the contract
            if raw.get(key) is not None:
                out[key] = raw[key].to(self.device)
        out["key"] = raw.get("key")
        if raw.get("hand_verts") is not None:
            raw = self.render_hands(raw)
        if raw.get("track_rgb") is None:
            return out

        out["track_clean"] = self.encode_pixels(raw["track_rgb"])
        cond = self.encode_pixels(self.condition_pixels(raw["head_rgb"], raw["head_depth_m"], raw["head_mask"]))
        out["rgb_condition"], out["depth_condition"], out["mask_condition"] = cond[:b], cond[b:2 * b], cond[2 * b:]
        present = torch.ones(b, 3, dtype=torch.bool, device=self.device)
        if raw.get("depth_present") is not None:
            present[:, 1] = raw["depth_present"].to(self.device).bool()
        out["condition_present"] = present

        lat_h, lat_w = out["track_clean"].shape[-2:]
        fg = raw["track_foreground"].to(self.device).bool()                                     # [B, 9, H, W]
        roles = raw["track_role_px"].to(self.device)                                           # [B, 9, H, W]
        groups = [(0, 1), (1, 5), (5, 9)]  # causal Wan VAE latent frames: anchor | transitions 1:5 | 5:9
        valid_groups = torch.stack([fg[:, a:z].any(dim=1) for a, z in groups], dim=1).float()
        out["track_valid"] = F.adaptive_max_pool2d(valid_groups, (lat_h, lat_w)).bool()[:, None]  # [B, 1, 3, h, w]
        role_groups = torch.stack([roles[:, a:z].max(dim=1).values for a, z in groups], dim=1)
        out["track_role"] = majority_pool(role_groups, (lat_h, lat_w))

        tok_h, tok_w = lat_h // self.config.token_patch, lat_w // self.config.token_patch
        delta = raw["track_delta"].to(self.device).float()                                     # [B, 8, H, W, 3]
        fg8 = fg[:, 1:].float()
        num = F.adaptive_avg_pool2d((delta * fg8[..., None]).permute(0, 1, 4, 2, 3).flatten(0, 1), (tok_h, tok_w))
        den = F.adaptive_avg_pool2d(fg8.flatten(0, 1)[:, None], (tok_h, tok_w)).clamp(min=1e-6)
        disp = (num / den).reshape(b, 8, 3, tok_h, tok_w).permute(0, 1, 3, 4, 2)
        cell_fg = F.adaptive_max_pool2d(fg8.flatten(0, 1)[:, None], (tok_h, tok_w)).reshape(b, 8, tok_h, tok_w) > 0
        out["track_disp_frames"] = torch.where(cell_fg[..., None], disp, torch.zeros_like(disp))
        role_frames = majority_pool(roles[:, 1:], (tok_h, tok_w))
        out["track_role_frames"] = torch.where(cell_fg, role_frames, torch.zeros_like(role_frames))
        out["camera_delta"] = raw["camera_delta"].to(self.device)
        out["camera_valid"] = raw["camera_valid"].to(self.device)
        return out


__all__ = ["HumanEncoderConfig", "HumanOnlineEncoder"]
