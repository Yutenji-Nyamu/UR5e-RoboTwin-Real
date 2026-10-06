"""Train and sample Action with a frozen native Video expert and current RGB only."""

import numpy as np
import torch

from .contract import HORIZON, SLOTS, absolute_actions, delta_actions, normalize, to_unified


class ActionPolicy:
    def __init__(self, model, stats):
        self.model, self.stats = model, stats

    def tensor(self, value):
        parameter = next(self.model.parameters())
        return torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)

    def inputs(self, latent, state, noisy, sigma):
        from metiswam4d.model import ModelInput

        batch = len(state)
        latent = self.tensor(latent)
        if (latent.ndim != 5 or latent.shape[:3] != (batch, self.model.video.config.in_channels, 1)
                or any(size % 2 for size in latent.shape[-2:]) or not torch.isfinite(latent).all()):
            raise ValueError("RGB condition must be finite [B, C, 1, even H, even W] current-frame latents")
        proprio = self.tensor(to_unified(normalize(state, self.stats, "state")))[:, None]
        mask = torch.zeros_like(proprio, dtype=torch.bool)
        mask[..., list(SLOTS)] = True
        return ModelInput(
            context=latent.new_empty(batch, 0, self.model.config.text_dim),
            video=latent, video_sigma=latent.new_zeros(batch),
            action=noisy, action_sigma=sigma, proprio=proprio, proprio_mask=mask,
        )

    def velocity(self, latent, state, noisy, sigma):
        parameter = next(self.model.parameters())
        # Trainable parameters stay FP32. Autocast supplies BF16 matmuls without
        # rounding away small AdamW updates in the parameters/optimizer state.
        with torch.autocast(device_type=parameter.device.type, dtype=torch.bfloat16,
                            enabled=parameter.dtype == torch.bfloat16):
            return self.model(self.inputs(latent, state, noisy, sigma)).action_velocity

    def loss(self, latent, state, actions, valid):
        target7 = normalize(delta_actions(state, actions), self.stats, "action")
        if target7.shape != (len(state), HORIZON, 7):
            raise ValueError("actions must be [B, 50, 7]")
        valid = self.tensor(valid).bool()
        if valid.shape != target7.shape[:2] or not valid.any() or not valid.any(dim=1).all():
            raise ValueError("each sample needs at least one valid action step")
        target = self.tensor(to_unified(target7))
        active = torch.zeros_like(target, dtype=torch.bool)
        active[..., list(SLOTS)] = valid[..., None]
        target = target * active
        noise = torch.randn_like(target) * active
        sigma = torch.rand(len(state), device=target.device, dtype=target.dtype)
        noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
        velocity = self.velocity(latent, state, noisy, sigma)
        # d[(1-sigma)*data + sigma*noise]/d sigma = noise-data.
        # Neither the 73 unused slots nor padded tail steps enter the loss.
        error = (velocity.float() - (noise - target).float()).square()
        return error[active].mean()

    @torch.no_grad()
    def predict(self, latent, state, *, rounds=16, seed=0):
        if not 1 <= rounds <= 1000:
            raise ValueError("sampling rounds must be in 1..1000")
        self.model.eval()
        parameter = next(self.model.parameters())
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
        action = torch.randn((len(state), HORIZON, 80), generator=generator,
                             device=parameter.device, dtype=parameter.dtype)
        active = torch.zeros_like(action)
        active[..., list(SLOTS)] = 1
        action *= active
        for step in range(rounds):
            sigma = action.new_full((len(state),), 1 - step / rounds)
            velocity = self.velocity(latent, state, action, sigma)
            action = (action - velocity / rounds) * active
        delta = normalize(action[..., list(SLOTS)].float().cpu().numpy(), self.stats, "action", inverse=True)
        return absolute_actions(state, delta)


class WanRgbEncoder:
    """Two RGB views: resize each to 320x192, stack head above wrist; no depth."""

    def __init__(self, path, *, device="cuda", dtype=torch.bfloat16):
        from diffusers import AutoencoderKLWan
        from pathlib import Path

        path = Path(path)
        if not (path / "config.json").is_file():
            raise FileNotFoundError("--vae must point to a local Wan VAE directory containing config.json")
        self.vae = AutoencoderKLWan.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
        self.vae.requires_grad_(False).eval().to(device)
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        if not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("invalid VAE latent scale")

    @torch.no_grad()
    def encode(self, head_rgb, wrist_rgb):
        import cv2

        views = []
        for view in (head_rgb, wrist_rgb):
            view = np.asarray(view)
            if view.ndim != 3 or view.shape[-1] != 3 or view.dtype != np.uint8:
                raise ValueError("encoder expects HWC uint8 RGB")
            views.append(cv2.resize(view, (320, 192), interpolation=cv2.INTER_AREA))
        rgb = np.concatenate(views, axis=0)
        pixels = torch.as_tensor(rgb.copy(), device=self.device, dtype=self.dtype).permute(2, 0, 1)[None, :, None]
        latent = self.vae.encode(pixels / 127.5 - 1).latent_dist.mode()
        return (latent - self.mean) / self.std
