"""Checkpoint panels for the Memory / Open model.

Same panels and scalars as ``metiswam4d.train.visualize`` on the current window (frame strip GT / sample, Track,
focus, action curves), plus a memory strip (the five memory frames + the current frame) and the deployment-path
action error ``vis/action_mse_deploy`` (``ActionSampler``: clean world only, 10 steps).  Sampling keeps the whole
six-frame clean prefix fixed.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from metiswam4d.data.contract import move_batch
from metiswam4d.sampler import AsyncSampler, SampleConditions, SampleResult
from metiswam4d.schedule import add_noise
from metiswam4d.train.visualize import Visualizer, frame_strip

from metiswam4d_inspired_by_internw0.data import MEMORY_SLOTS
from metiswam4d_inspired_by_internw0.model import CLEAN_PREFIX, ActionSampler


class PrefixSampler(AsyncSampler):
    """``AsyncSampler`` whose Video clean prefix is ``CLEAN_PREFIX`` latent frames (memory + current)."""

    @torch.no_grad()
    def sample(self, model, cond: SampleConditions, *, generator=None, on_action_ready=None, stop_after=None,
               dtype=None) -> SampleResult:
        from metiswam4d.model import ModelInput
        present = cond.present
        schedule = self.schedule.present(present)
        trajectory = schedule.inference_trajectory(self.rounds)
        device, b = cond.context.device, cond.context.shape[0]
        dtype = dtype or cond.context.dtype

        def noise(shape) -> Tensor:
            return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)

        x: dict[str, Tensor] = {}
        prefix = {"video": cond.video_first_frame.shape[2] if "video" in present else 0, "track": 1}
        if "video" in present:
            c, _, h, w = cond.video_first_frame.shape[1:]
            x["video"] = torch.cat((cond.video_first_frame.to(dtype), noise((b, c, cond.video_future_frames, h, w))), 2)
        if "track" in present:
            c, _, h, w = cond.track_anchor.shape[1:]
            x["track"] = torch.cat((cond.track_anchor.to(dtype), noise((b, c, cond.track_future_frames, h, w))), 2)
            if cond.camera_frames > 0:
                x["camera"] = noise((b, cond.camera_frames, cond.camera_dim))
        if "action" in present:
            x["action"] = noise((b, cond.action_horizon, cond.action_dim))
        done: dict[str, int] = {}
        rounds_run = 0
        for i in range(self.rounds):
            now = {m: trajectory[m][i].item() for m in present}
            nxt = {m: trajectory[m][i + 1].item() for m in present}
            if all(now[m] <= 0 for m in present):
                break
            out = model(ModelInput(
                context=cond.context, context_mask=cond.context_mask,
                video=x.get("video"), video_sigma=self._sigma(now, "video", b, device),
                track=x.get("track"), track_sigma=self._sigma(now, "track", b, device),
                track_conditions=cond.track_conditions if "track" in present else None,
                condition_present=cond.condition_present if "track" in present else None, camera=x.get("camera"),
                action=x.get("action"), action_sigma=self._sigma(now, "action", b, device),
                proprio=cond.proprio, proprio_mask=cond.proprio_mask, embodiment=cond.embodiment))
            rounds_run = i + 1
            for m in present:
                if now[m] <= 0:
                    continue
                v = out.velocity(m).to(x[m].dtype)
                step = nxt[m] - now[m]
                if m in prefix:
                    k = prefix[m]
                    x[m][:, :, k:] = x[m][:, :, k:] + step * v[:, :, k:]
                    if m == "track" and "camera" in x and out.camera_velocity is not None:
                        x["camera"] = x["camera"] + step * out.camera_velocity.to(x["camera"].dtype)
                else:
                    x[m] = x[m] + step * v
                if nxt[m] <= 0 and m not in done:
                    done[m] = i + 1
        return SampleResult(x.get("video"), x.get("track"), x.get("action"), done, rounds_run, camera=x.get("camera"))


def _window_view(batch: dict) -> dict:
    view = dict(batch)
    view["video_clean"] = batch["video_clean"][:, :, MEMORY_SLOTS:]
    return view


class IW0Visualizer(Visualizer):
    def __init__(self, *args, schedule, **kwargs):
        super().__init__(*args, schedule=schedule, **kwargs)
        self.sampler = PrefixSampler(schedule, self.rounds)
        self.action_sampler = ActionSampler(schedule, steps=self.rounds)

    @torch.no_grad()
    def _focus_forward(self, model, batch: dict):
        from metiswam4d.model import ModelInput
        g = torch.Generator().manual_seed(11)
        b = batch["text_context"].shape[0]

        def noisy(x: Tensor, s: float, keep: int) -> Tensor:
            out, _, _ = add_noise(x, torch.full((b,), s, device=x.device), torch.randn(x.shape, generator=g).to(x))
            out[:, :, :keep] = x[:, :, :keep]
            return out

        def sig(s: float) -> Tensor:
            return torch.full((b,), s, device=self.device)

        inputs = ModelInput(
            context=batch["text_context"], context_mask=batch.get("text_mask"),
            video=noisy(batch["video_clean"], 0.5, CLEAN_PREFIX), video_sigma=sig(0.5),
            track=noisy(batch["track_clean"], 0.3, 1), track_sigma=sig(0.3),
            track_conditions=tuple(batch[k] for k in ("rgb_condition", "depth_condition", "mask_condition")),
            camera=noisy(batch["camera_delta"], 0.3, 0) if batch.get("camera_delta") is not None else None,
            action=noisy(batch["action"], 0.2, 0), action_sigma=sig(0.2),
            proprio=batch.get("proprio"), proprio_mask=batch.get("proprio_mask"), embodiment=batch.get("embodiment"))
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = model(inputs)
        return out.focus, {"progress_video": out.progress_video, "progress_body": out.progress_body,
                           "progress_action": out.progress_action}

    def _run_batch(self, model, batch: dict, step: int, source: str) -> dict[str, float]:
        f = self.spec.track_frames
        cond = SampleConditions(
            context=batch["text_context"], context_mask=batch.get("text_mask"),
            video_first_frame=batch["video_clean"][:, :, :CLEAN_PREFIX], video_future_frames=f - 1,
            track_anchor=batch["track_clean"][:, :, :1], track_future_frames=f - 1,
            track_conditions=tuple(batch[k] for k in ("rgb_condition", "depth_condition", "mask_condition")),
            camera_frames=self.spec.frame_slots if batch.get("camera_delta") is not None else 0,
            action_horizon=self.spec.action_horizon, action_dim=self.spec.action_dim,
            proprio=batch.get("proprio"), proprio_mask=batch.get("proprio_mask"), embodiment=batch.get("embodiment"))
        generator = torch.Generator().manual_seed(self.seed)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            result = self.sampler.sample(model, cond, generator=generator, dtype=self.dtype)
            deploy = self.action_sampler.sample(
                model, context=batch["text_context"], context_mask=batch.get("text_mask"),
                video_clean=batch["video_clean"][:, :, :CLEAN_PREFIX], track_anchor=batch["track_clean"][:, :, :1],
                track_conditions=tuple(batch[k] for k in ("rgb_condition", "depth_condition", "mask_condition")),
                proprio=batch["proprio"], proprio_mask=batch.get("proprio_mask"), embodiment=batch["embodiment"],
                action_horizon=self.spec.action_horizon, action_dim=self.spec.action_dim,
                generator=torch.Generator().manual_seed(self.seed))
        focus, progress = self._focus_forward(model, batch)
        view = _window_view(batch)
        window = replace(result, video=result.video[:, :, MEMORY_SLOTS:])
        scalars = self._scalars(view, window)
        m = batch["action_mask"].float()
        scalars["vis/action_mse_deploy"] = float(((deploy.float() - batch["action"].float()).square() * m).sum()
                                                  / m.sum().clamp(min=1.0))
        for name, logits in progress.items():
            label = batch.get("progress_body" if name == "progress_action" else name)
            if logits is not None and label is not None:
                scalars[f"vis/{name}_abs_err"] = float((torch.sigmoid(logits.float()).reshape(-1)
                                                        - label.float().reshape(-1)).abs().mean())
        if self.is_main:
            self._write(view, replace(window, action=deploy), step, scalars, focus, source)
            self._write_memory(batch, step, source)
        return scalars

    def _write_memory(self, batch: dict, step: int, source: str) -> None:
        from PIL import Image
        if self.encoder is None:
            return
        folder = self.out_dir / "vis" / f"step_{step:07d}"
        prefix = batch["video_clean"][:, :, :CLEAN_PREFIX]   # independent single-frame latents
        frames = torch.cat([self.encoder.decode_latents(prefix[:, :, k:k + 1]) for k in range(CLEAN_PREFIX)], 1).cpu()
        keys = batch.get("key") or [str(i) for i in range(frames.shape[0])]
        for i, key in enumerate(keys):
            strip = frame_strip([frames[i]])
            Image.fromarray(strip).save(folder / f"{source + '_' if source else ''}{i}_{str(key).replace('/', '_')}_memory.png")
            if self.tb is not None:
                self.tb.add_image(f"vis/memory_{i}", strip, step, dataformats="HWC")


def build_iw0_visualizer(stage, *, encoder, device, dtype, rank: int, world: int, out_dir: Path, is_main: bool, tb,
                         schedule, log=print):
    tcfg = stage.training
    if not tcfg.visualize:
        return None
    from metiswam4d.data.robodojo import window_dataset
    from metiswam4d.data.rt2 import collate_raw
    rdj = stage.data.robodojo
    dataset = window_dataset(replace(rdj, split=rdj.val_split or rdj.split, fixed_windows=True))
    g = torch.Generator().manual_seed(1234)
    fixed = int(torch.randperm(len(dataset), generator=g)[rank % len(dataset)])
    n = max(1, int(tcfg.visualize_samples))
    log(f"[vis] per rank: 1 fixed {dataset.config.split} window + {n - 1} random per checkpoint, "
        f"{tcfg.visualize_rounds} rounds (prefix sampler) + deployment-path action sampler")
    return IW0Visualizer(raw_batch=None, dataset=dataset, collate=collate_raw, fixed_index=fixed, spec=stage.data.spec,
                         schedule=schedule, rounds=tcfg.visualize_rounds, encoder=encoder, device=device, dtype=dtype,
                         out_dir=out_dir, is_main=is_main, tb=tb, seed=1234 + rank, samples=n)


__all__ = ["IW0Visualizer", "PrefixSampler", "build_iw0_visualizer"]
