"""Probe harness for the OpenWAM-Alpha Sim-RoboTwin-Full baseline (Video + Action, no Track).

Same interface as :class:`harness.JanusProbe` (``dataset``, ``predict_window``, ``action_errors``)
so ``run_openloop.py`` can drive both.  The joint attention is hooked at
``DualSystemMoTDriver._mixed_attention`` (one SDPA over ``[V_clean | V_future | A]`` per layer);
segments are reported with empty Track ranges so the generic interventions apply unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from harness import JANUS_ROOT, Intervention, JanusProbe, ProbeAttention, Segments
from janusact4d_rt2imperfect_v1.data import ImperfectRoboTwinDataset
from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats
from openwam.deploy.denoise_schedule import schedule_sync
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

ALPHA_DIR = Path("/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full")


class AlphaAttention(ProbeAttention):
    """Adapter: ``_mixed_attention(q_cat, k_cat, v_cat, attn_mask)`` with ``(B, S, H*D)`` tensors."""

    def __call__(self, q_cat, k_cat, v_cat, attn_mask=None):
        if attn_mask is None:
            attn_mask = torch.ones(q_cat.shape[1], k_cat.shape[1], dtype=torch.bool, device=q_cat.device)
        return super().__call__(q_cat, k_cat, v_cat, attn_mask, heads=self.heads)


class AlphaProbe:
    def __init__(self, alpha_dir: str | Path = ALPHA_DIR, denoise_steps: int = 10, config: dict | None = None):
        import yaml
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.steps = denoise_steps
        torch.set_num_threads(4)
        config = config or yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
        self.config = config
        print("Building Alpha architecture (RoboTwin-Full safetensors)", flush=True)
        _, alpha, _ = build_architecture_from_ckpt_dir(str(alpha_dir), weights_required=True)
        alpha.requires_grad_(False).eval()
        alpha.set_dtype_device(self.dtype, self.device)
        self.alpha = alpha
        driver = alpha._mot_driver or alpha.build_mot_driver()
        self.attn = AlphaAttention(alpha.video_backbone.num_heads)
        driver._mixed_attention = self.attn                    # instance-level patch
        self.normalizer = Normalizer(mode="min_max", stats=load_mode_stats(str(Path(alpha_dir) / "normalization_stats.npy"), "eef"))
        self.dataset = ImperfectRoboTwinDataset(config["data_root"], Path(alpha_dir) / "normalization_stats.npy")
        self.text_cache = {}
        self.segments: Segments | None = None
        print("MODEL_READY", flush=True)

    def _measure_segments(self, inputs) -> Segments:
        lat = inputs["latents"]
        f, h, w = lat.shape[2], lat.shape[3] // 2, lat.shape[4] // 2
        v_clean, v_all = h * w, f * h * w
        ranges = {"video_clean": (0, v_clean), "video_future": (v_clean, v_all),
                  "track_cond": (v_all, v_all), "track_anchor": (v_all, v_all), "track_future": (v_all, v_all),
                  "action": (v_all, v_all + 32)}
        return Segments(ranges, (f, h, w), (1, 0, 0))

    @torch.inference_mode()
    def predict_window(self, sample: dict, seed: int, *, intervention: Intervention | None = None,
                       capture_attn: bool = False, capture_kv: bool = False, static_future: bool = False,
                       decode: bool = False) -> dict:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        vb, ab = self.alpha.video_backbone, self.alpha.action_backbone
        inputs = vb.preprocess_input_for_inference(
            prompt=sample["prompt"], first_frame_image=list(sample["first_frame_image"]),
            num_frames=9, height=384, width=320, seed=seed,
            num_inference_steps=self.steps, shift=vb.shift_video, tiled=False,
            prompt_embed_cache=self.text_cache)
        ref = inputs["first_frame_latents"]
        inputs["latents"][:, :, :1] = ref
        action_noise = torch.randn((1, 32, 80), device=self.device, dtype=self.dtype)
        action = action_noise.clone()
        active = sample["proprio_mask"][0].to(self.device)
        proprio = sample["proprio"].to(self.device, self.dtype)[None]

        self.segments = self.attn.segments = self._measure_segments(inputs)
        self.attn.intervention = intervention or Intervention.none()
        self.attn.capture_attn, self.attn.capture_kv = capture_attn, capture_kv
        self.attn.kv_store = {}
        attn_steps = []
        schedule = schedule_sync(vb.scheduler, ab.scheduler, self.steps, shift=ab.shift_action, shift_video=vb.shift_video)
        for step, ((tv, ta), (tv_next, ta_next)) in enumerate(zip(schedule[:-1], schedule[1:])):
            sv, sa = tv / vb.scheduler.num_train_timesteps, ta / ab.scheduler.num_train_timesteps
            svn, san = tv_next / vb.scheduler.num_train_timesteps, ta_next / ab.scheduler.num_train_timesteps
            if static_future:
                inputs["latents"][:, :, 1:] = ref
            call = {k: v for k, v in inputs.items()}
            call["timestep"] = torch.tensor([tv], device=self.device, dtype=self.dtype)
            call["_proprio_sample_mask"] = active[None, None]
            self.attn.reset_forward(step)
            video_v, action_v = self.alpha(action, torch.tensor([ta], device=self.device, dtype=self.dtype),
                                           proprio=proprio, **call)
            inputs["latents"] = inputs["latents"] + video_v * (svn - sv)
            inputs["latents"][:, :, :1] = ref
            action = ab.scheduler.flow_step(action_v, sa, san, action)
            action[..., ~active] = action_noise[..., ~active] * san
            if capture_attn:
                layers = torch.stack(self.attn.attn_layers)
                attn_steps.append(dict(by_head=layers[:, 0].sum(dim=2).cpu(), by_query=layers[:, 0].mean(dim=1).cpu()))
        values = action[0].float().cpu().numpy()
        out = dict(action_norm=values, segments=self.segments, video_latents=inputs["latents"], track_latents=None)
        if capture_attn:
            out["attn_by_head"] = torch.stack([s["by_head"] for s in attn_steps]).numpy()
            out["attn_by_query"] = torch.stack([s["by_query"] for s in attn_steps]).numpy()
        if capture_kv:
            out["kv"] = self.attn.kv_store
        if decode:
            out["pred_rgb"] = np.stack([np.asarray(x) for x in vb.decode_video(inputs["latents"], tiled=False)])
            out["pred_track"] = np.zeros((0,), dtype=np.uint8)
        return out

    action_errors = JanusProbe.action_errors


__all__ = ["AlphaProbe", "AlphaAttention"]
