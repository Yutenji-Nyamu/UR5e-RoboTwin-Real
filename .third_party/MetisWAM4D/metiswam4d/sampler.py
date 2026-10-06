"""Multi-rate asynchronous sampler.

All present modalities start from noise and follow the shared progress axis of
:class:`~metiswam4d.schedule.AsyncSchedule`.  Every round evaluates the joint
velocity once and updates the modalities that have not finished:

    x_m <- x_m + (sigma_m^next - sigma_m) * v_m .

A modality whose sigma reaches zero is frozen and enters later rounds as a
clean condition (timestep 0).  With ``rounds = 20`` and completion positions
``r_A = r_T = 0.5, r_V = 1`` Track and Action are denoised together and are
final after round 10, Video after round 20; ``on_action_ready`` is invoked as
soon as the action block is final so control can be dispatched before the
video finishes (``stop_after="action"`` skips the remaining video rounds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from metiswam4d.model import MetisWAM4D, ModelInput
from metiswam4d.schedule import AsyncSchedule


@dataclass
class SampleConditions:
    context: Tensor
    context_mask: Tensor | None = None
    video_first_frame: Tensor | None = None      # [B, C, 1, H, W] clean current observation latent
    video_future_frames: int = 0                 # latent future frames to generate
    track_anchor: Tensor | None = None           # [B, C, 1, H_t, W_t] zero-displacement anchor latent
    track_future_frames: int = 0
    track_conditions: tuple[Tensor, ...] | None = None
    condition_present: Tensor | None = None      # [B, 3] bool, see ModelInput
    camera_frames: int = 0                       # camera ego-motion tokens to generate on the Track clock (N_f or 0)
    camera_dim: int = 9
    action_horizon: int = 0
    action_dim: int = 0
    proprio: Tensor | None = None
    proprio_mask: Tensor | None = None
    embodiment: Tensor | None = None             # [B] long registry index

    @property
    def present(self) -> tuple[str, ...]:
        names = []
        if self.video_first_frame is not None and self.video_future_frames > 0:
            names.append("video")
        if self.track_anchor is not None and self.track_future_frames > 0:
            names.append("track")
        if self.action_horizon > 0:
            names.append("action")
        return tuple(names)


@dataclass
class SampleResult:
    video: Tensor | None
    track: Tensor | None
    action: Tensor | None
    completion_round: dict[str, int]
    rounds_run: int
    camera: Tensor | None = None                 # [B, N_f, 9] camera codes (see data/camera.py)


class AsyncSampler:
    def __init__(self, schedule: AsyncSchedule, rounds: int = 16):
        if rounds < 1:
            raise ValueError("rounds must be positive")
        self.schedule = schedule
        self.rounds = rounds

    @torch.no_grad()
    def sample(
        self,
        model: MetisWAM4D,
        cond: SampleConditions,
        *,
        generator: torch.Generator | None = None,
        on_action_ready: Callable[[Tensor], None] | None = None,
        stop_after: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SampleResult:
        present = cond.present
        if not present:
            raise ValueError("nothing to sample")
        schedule = self.schedule.present(present)
        trajectory = schedule.inference_trajectory(self.rounds)
        device = cond.context.device
        b = cond.context.shape[0]
        dtype = dtype or cond.context.dtype

        def noise(shape) -> Tensor:
            return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)

        x: dict[str, Tensor] = {}
        if "video" in present:
            c, _, h, w = cond.video_first_frame.shape[1:]
            x["video"] = torch.cat((cond.video_first_frame.to(dtype),
                                    noise((b, c, cond.video_future_frames, h, w))), dim=2)
        if "track" in present:
            c, _, h, w = cond.track_anchor.shape[1:]
            x["track"] = torch.cat((cond.track_anchor.to(dtype),
                                    noise((b, c, cond.track_future_frames, h, w))), dim=2)
            if cond.camera_frames > 0:
                x["camera"] = noise((b, cond.camera_frames, cond.camera_dim))
        if "action" in present:
            x["action"] = noise((b, cond.action_horizon, cond.action_dim))

        done: dict[str, int] = {}
        rounds_run = 0
        for i in range(self.rounds):
            sigma_now = {m: trajectory[m][i].item() for m in present}
            sigma_next = {m: trajectory[m][i + 1].item() for m in present}
            if all(sigma_now[m] <= 0 for m in present):
                break
            inputs = ModelInput(
                context=cond.context, context_mask=cond.context_mask,
                video=x.get("video"), video_sigma=self._sigma(sigma_now, "video", b, device),
                track=x.get("track"), track_sigma=self._sigma(sigma_now, "track", b, device),
                track_conditions=cond.track_conditions if "track" in present else None,
                condition_present=cond.condition_present if "track" in present else None,
                camera=x.get("camera"),
                action=x.get("action"), action_sigma=self._sigma(sigma_now, "action", b, device),
                proprio=cond.proprio, proprio_mask=cond.proprio_mask, embodiment=cond.embodiment,
            )
            out = model(inputs)
            rounds_run = i + 1
            for m in present:
                if sigma_now[m] <= 0:
                    continue
                v = out.velocity(m).to(x[m].dtype)
                step = sigma_next[m] - sigma_now[m]
                if m == "video":
                    x[m][:, :, 1:] = x[m][:, :, 1:] + step * v[:, :, 1:]
                elif m == "track":
                    x[m][:, :, 1:] = x[m][:, :, 1:] + step * v[:, :, 1:]
                    if "camera" in x and out.camera_velocity is not None:  # camera tokens ride the Track clock
                        x["camera"] = x["camera"] + step * out.camera_velocity.to(x["camera"].dtype)
                else:
                    x[m] = x[m] + step * v
                if sigma_next[m] <= 0 and m not in done:
                    done[m] = i + 1
                    if m == "action" and on_action_ready is not None:
                        on_action_ready(x[m])
            if stop_after is not None and stop_after in done:
                break
        return SampleResult(x.get("video"), x.get("track"), x.get("action"), done, rounds_run, camera=x.get("camera"))

    @staticmethod
    def _sigma(values: dict[str, float], name: str, b: int, device) -> Tensor | None:
        if name not in values:
            return None
        return torch.full((b,), float(values[name]), device=device)


__all__ = ["AsyncSampler", "SampleConditions", "SampleResult"]
