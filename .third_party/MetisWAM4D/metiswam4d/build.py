"""Build a MetisWAM4D model (and its initialisation) from a StageConfig."""
from __future__ import annotations

from pathlib import Path

import torch

from metiswam4d.config import ExpertSize, ModelConfig, StageConfig
from metiswam4d.experts.alpha import (
    AlphaVideoSource, load_alpha_action, load_alpha_proprio, load_alpha_video,
)
from metiswam4d.experts.init import interpolate_from_source
from metiswam4d.experts.local import (
    ActionExpert, ActionExpertConfig, ExpertCoreConfig, LatentExpert, LatentExpertConfig,
    TrackExpert, TrackExpertConfig,
)
from metiswam4d.model import MetisWAM4D, MetisWAM4DConfig
from metiswam4d.schedule import AsyncSchedule, NoiseMixture


def _core(size: ExpertSize, text_dim: int, macro: int | None) -> ExpertCoreConfig:
    return ExpertCoreConfig(dim=size.dim, ffn_dim=size.ffn_dim, num_layers=size.layers, num_heads=size.heads,
                            head_dim=size.head_dim, text_dim=text_dim, freq_dim=size.freq_dim, macro_layers=macro)


def build_model(cfg: ModelConfig) -> MetisWAM4D:
    present = tuple(cfg.experts)
    macro_candidates = [getattr(cfg, n).layers for n in ("video", "action") if n in present]
    macro = max(macro_candidates) if macro_candidates else cfg.track.layers
    video = track = action = None
    if "video" in present:
        video = LatentExpert(LatentExpertConfig(
            core=_core(cfg.video, cfg.text_dim, macro), in_channels=cfg.latent_channels,
            out_channels=cfg.latent_channels, patch_size=tuple(cfg.patch)))
    if "track" in present:
        track = TrackExpert(TrackExpertConfig(
            core=_core(cfg.track, cfg.text_dim, macro), channels=cfg.latent_channels, patch_size=tuple(cfg.patch)))
    if "action" in present:
        action = ActionExpert(ActionExpertConfig(core=_core(cfg.action, cfg.text_dim, macro), action_dim=cfg.action_dim))
    focus = cfg.focus
    if focus is not None:
        focus.text_dim = cfg.text_dim
        focus.proprio_dim = cfg.proprio_dim
    action_read = cfg.action_read if action is not None else "none"
    return MetisWAM4D(video=video, track=track, action=action, config=MetisWAM4DConfig(
        action_read=action_read, focus=focus, text_dim=cfg.text_dim, proprio_dim=cfg.proprio_dim,
        num_embodiments=cfg.num_embodiments, gradient_checkpointing=cfg.gradient_checkpointing, dense_read_floor=cfg.dense_read_floor,
        read_action_horizon=cfg.read_action_horizon, read_action_dim=cfg.action_dim))


@torch.no_grad()
def initialize_model(model: MetisWAM4D, stage: StageConfig, *, log=print) -> dict:
    """Apply the configured warm start.  Order: Alpha -> Track init -> initialize_from."""
    init = stage.init
    report: dict = {}
    if init.alpha_checkpoint:
        if model.video is not None and init.load_video_from_alpha:
            report["alpha_video"] = load_alpha_video(model.video, init.alpha_checkpoint)
            log(f"[init] Alpha video: {report['alpha_video']['loaded']} tensors")
        if model.action is not None and init.load_action_from_alpha:
            report["alpha_action"] = load_alpha_action(model.action, init.alpha_checkpoint)
            log(f"[init] Alpha action: {report['alpha_action']['loaded']} tensors")
            report["alpha_proprio"] = load_alpha_proprio(model.proprio_encoder, init.alpha_checkpoint)
    if model.track is not None:
        mode = init.track.mode
        if mode == "interpolate_from_video":
            if model.video is not None and init.alpha_checkpoint is None:
                source = {k: v for k, v in model.video.state_dict().items()}
                report["track_init"] = interpolate_from_source(model.track, source)
            elif init.alpha_checkpoint:
                source = AlphaVideoSource(init.alpha_checkpoint)
                try:
                    report["track_init"] = interpolate_from_source(model.track, source, source_prefix="video_backbone.dit.")
                finally:
                    source.close()
            else:
                raise ValueError("track interpolation needs a Video expert or an Alpha checkpoint")
            r = report["track_init"]
            log(f"[init] Track interpolated: {len(r['loaded'])} copied, {len(r['resized'])} resized, "
                f"{len(r['fresh'])} fresh, {len(r['skipped'])} skipped")
        elif mode == "v3_dcp":
            if not init.track.path:
                raise ValueError("track init mode=v3_dcp requires the DCP step directory")
            from metiswam4d.experts.legacy import load_v3_track
            report["track_init"] = load_v3_track(model.track, init.track.path)
            log(f"[init] Track from JanusTrack v3 DCP: {report['track_init']['loaded']} tensors")
        elif mode == "checkpoint":
            if not init.track.path:
                raise ValueError("track init mode=checkpoint requires a path")
            state = torch.load(init.track.path, map_location="cpu", weights_only=True)
            state = state.get("track", state)
            missing, unexpected = model.track.load_state_dict(state, strict=False)
            report["track_init"] = {"missing": missing, "unexpected": unexpected}
            log(f"[init] Track checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
        elif mode != "fresh":
            raise ValueError(f"unknown track init mode {mode!r}")
    if init.initialize_from:
        from metiswam4d.train.checkpoint import load_model_weights
        prefixes = tuple(init.initialize_from_prefixes) if init.initialize_from_prefixes else None
        exclude = tuple(init.initialize_from_exclude) if init.initialize_from_exclude else None
        report["initialize_from"] = load_model_weights(model, init.initialize_from, prefixes=prefixes, exclude=exclude)
        log(f"[init] weights from {init.initialize_from} (prefixes={prefixes}, exclude={exclude}): {report['initialize_from']}")
    return report


def build_schedule(stage: StageConfig) -> tuple[AsyncSchedule, NoiseMixture]:
    s = stage.schedule
    present = tuple(stage.model.experts)
    schedule = AsyncSchedule({k: v for k, v in s.completion.items() if k in present},
                             {k: v for k, v in s.shift.items() if k in present})
    return schedule, NoiseMixture(dict(s.mixture))


__all__ = ["build_model", "build_schedule", "initialize_model"]
