"""Probe harness for FlowWAM (official ``flowwam_robotwin.safetensors``).

FlowWAM is two-stage: (1) a dual-stream Wan2.2 DiT denoises RGB + optical-flow latents of the T-tiled
three-camera image (25 steps); (2) the DiT is re-run once at t=0 and all 30 layers' ``[rgb, flow]`` tokens are
captured; (3) a separate 30-layer action DiT denoises a 33-step qpos chunk (50 steps), block *i* cross-attending
to captured layer *i*.  Stage 3 is the only place where actions read the world, so all interventions are applied
to the captured feature list (the video prediction is untouched by construction) and the attention weights are
read from ``ActionVideoCrossAttention``.

Stages 1-2 are cached per (window, seed, static) so every condition costs only the action DiT.

Key layout per captured layer (720 tokens): ``video_clean`` = rgb latent frame 0 [0,120), ``video_future`` =
rgb frames 1-2 [120,360), ``flow_clean`` [360,480), ``flow_future`` [480,720).  ``video_grid`` = (3, 12, 10)
(same T-layout as OpenWAM-Alpha).  Generic interventions on ``video_future`` are applied to both the rgb and the
flow future tokens; ``flow_future`` alone can be targeted explicitly.

Actions are absolute 14-D joint positions (z-scored); errors are reported in joint degrees (``joint_deg``) and
gripper units, ``pos_cm`` is NaN (no FK here).
"""
from __future__ import annotations

import sys
import types
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

FLOW_ROOT = Path("/m2v_intern_v3/danglingwei/codes/FlowWAM_260824")
CKPT = Path("/ytech_face_algo_ssd/danglingwei/model_zoos/YixiangChen/FlowWAM/flowwam_robotwin.safetensors")
NORM = CKPT.with_name("flowwam_robotwin_action_norm_stats.npz")
LOCAL_MODELS = "/m2v_intern/_public_models"
for p in (str(FLOW_ROOT), str(FLOW_ROOT / "inference")):
    if p not in sys.path:
        sys.path.insert(0, p)
if "msgpack_numpy" not in sys.modules:                    # only used by the websocket protocol
    stub = types.ModuleType("msgpack_numpy")
    stub.patch = lambda: None
    sys.modules["msgpack_numpy"] = stub

from harness import Intervention, Segments  # noqa: E402
from janusact4d_rt2imperfect_v1.data import ImperfectRoboTwinDataset, decode_source_jpeg  # noqa: E402
from openwam.dataloader.transforms.multiview import format_prompt_for_inference  # noqa: E402

import action_dit as _action_dit  # noqa: E402
import flow_action_server as S  # noqa: E402
from einops import rearrange  # noqa: E402

CAMERAS = ("head_camera", "left_camera", "right_camera")
VIDEO_FRAMES, NUM_FRAMES = 9, 33
VIDEO_STEPS, ACTION_STEPS = 25, 50
SIGMA_SHIFT, ACTION_SNR_SHIFT, ACTION_COND_SIGMA = 5.0, 5.0, 0.0
SIZE = (320, 256)


class CrossAttnProbe:
    """State shared with the patched ``ActionVideoCrossAttention.forward``."""

    def __init__(self):
        self.capture = False
        self.layer = -1
        self.probs: list[torch.Tensor] = []

    def reset(self):
        self.layer = -1
        self.probs = []


_PROBE = CrossAttnProbe()
_orig_forward = _action_dit.ActionVideoCrossAttention.forward


def _patched_forward(self, x, ctx):
    _PROBE.layer += 1
    if not _PROBE.capture:
        return _orig_forward(self, x, ctx)
    q = rearrange(self.norm_q(self.q(x)), "b s (n d) -> b n s d", n=self.num_heads)
    k = rearrange(self.norm_k(self.k(ctx)), "b s (n d) -> b n s d", n=self.num_heads)
    v = rearrange(self.v(ctx), "b s (n d) -> b n s d", n=self.num_heads)
    logits = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    probs = logits.float().softmax(-1)
    _PROBE.probs.append(probs.detach())
    out = torch.matmul(probs.to(v.dtype), v)
    return self.o(rearrange(out, "b n s d -> b s (n d)"))


_action_dit.ActionVideoCrossAttention.forward = _patched_forward


class _WindowDataset:
    def __init__(self, base: ImperfectRoboTwinDataset):
        self.base, self.rows, self.instructions = base, base.rows, base.instructions

    def read_window(self, row_index: int, start: int) -> dict:
        import random
        row = self.rows[int(row_index)]
        with h5py.File(row["source"], "r") as src:
            total = int(src["observation/head_camera/depth"].shape[0])
            if start + 32 >= total:
                raise ValueError("window exceeds episode")
            frames = {c: np.asarray(decode_source_jpeg(src[f"observation/{c}/rgb"][start])) for c in CAMERAS}
            qpos = src["joint_action/vector"][start:start + 33].astype(np.float32)          # [33, 14]
        pool = self.instructions[int(row_index)].get("seen") or self.instructions[int(row_index)].get("unseen")
        instruction = random.choice(pool)
        return dict(prompt=format_prompt_for_inference(instruction), instruction=instruction, frames=frames,
                    qpos0=qpos[0], action=torch.from_numpy(qpos[1:]), key=f"{row['task']}/{row['variant']}/episode{row['episode']}/f{start}",
                    bucket=row["bucket"], head_rgb=torch.from_numpy(frames["head_camera"].copy()))


class FlowWAMProbe:
    def __init__(self):
        import yaml
        from harness import JANUS_ROOT
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.steps = ACTION_STEPS
        print("Building FlowWAM (dual-stream Wan2.2 + ActionExpertIDM)", flush=True)
        self.pipe, self.flow_stream = S.build_pipeline(local_model_path=LOCAL_MODELS, device=self.device, full_path=str(CKPT))
        self.action_expert = S.load_action_expert_idm(
            checkpoint_path=str(CKPT), action_dim=14, num_frames=NUM_FRAMES, text_context_dim=4096, video_dim=int(self.pipe.dit.dim),
            device=self.device, joint_state_dim=14, num_action_layers=30, action_expert_dim=1024, action_expert_heads=16,
            action_expert_ffn_dim=4096, pred_target="velocity", use_rope=True, proprio_mode="text")
        self.action_norm = S.ActionNormStats.load(str(NORM))
        janus_cfg = yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
        self.dataset = _WindowDataset(ImperfectRoboTwinDataset(janus_cfg["data_root"], Path(janus_cfg["alpha_checkpoint"]) / "normalization_stats.npy"))
        self.stage_cache: OrderedDict = OrderedDict()
        self.text_cache: dict = {}
        self.segments: Segments | None = None
        print("MODEL_READY", flush=True)

    # -- stages 1-2 -------------------------------------------------------------------------
    @torch.inference_mode()
    def _stage12(self, sample: dict, seed: int, static: bool) -> dict:
        key = (sample["key"], sample["instruction"], seed, static)
        if key in self.stage_cache:
            self.stage_cache.move_to_end(key)
            return self.stage_cache[key]
        pipe, dtype, device = self.pipe, self.dtype, self.device
        w, h = SIZE
        from PIL import Image
        imgs = {c: np.array(Image.fromarray(sample["frames"][c]).resize((w, h), Image.BICUBIC)) for c in CAMERAS}
        tiled = S.tshape_tile(imgs["head_camera"], imgs["left_camera"], imgs["right_camera"])
        th, tw, vf = pipe.check_resize_height_width(tiled.shape[0], tiled.shape[1], VIDEO_FRAMES)
        tiled_pil = Image.fromarray(tiled).resize((tw, th), Image.BICUBIC)
        instr = sample["instruction"]
        if instr not in self.text_cache:
            pipe.load_models_to_device(["text_encoder"])
            self.text_cache[instr] = (pipe.prompter.encode_prompt(S.RoboTwinActionFlowDataset.CAMERA_PREFIX + instr, positive=True, device=device),
                                      pipe.prompter.encode_prompt(instr, positive=True, device=device))
        context, action_context = self.text_cache[instr]
        pipe.load_models_to_device(["vae"])
        up = pipe.vae.upsampling_factor
        T_lat = (vf - 1) // 4 + 1
        rgb_prefix = pipe.vae.encode(pipe.preprocess_video([tiled_pil]), device=device).to(dtype=dtype, device=device)
        flow_prefix = pipe.vae.encode(pipe.preprocess_video([Image.new("RGB", (tw, th), (255, 255, 255))]), device=device).to(dtype=dtype, device=device)
        z = getattr(pipe.vae, "z_dim", 16)
        shape = (1, z, T_lat, th // up, tw // up)
        rgb_latents = pipe.generate_noise(shape, seed=seed, rand_device="cpu").to(dtype=dtype, device=device)
        flow_latents = pipe.generate_noise(shape, seed=seed + 1, rand_device="cpu").to(dtype=dtype, device=device)
        rgb_latents[:, :, :1], flow_latents[:, :, :1] = rgb_prefix, flow_prefix
        pipe.load_models_to_device(pipe.in_iteration_models)
        if static:
            rgb_latents = rgb_prefix.expand(-1, -1, T_lat, -1, -1).clone()
            flow_latents = flow_prefix.expand(-1, -1, T_lat, -1, -1).clone()
        else:
            pipe.scheduler.set_timesteps(VIDEO_STEPS, shift=SIGMA_SHIFT)
            for i, timestep in enumerate(pipe.scheduler.timesteps):
                t = timestep.unsqueeze(0).to(dtype=dtype, device=device)
                rgb_pred, flow_pred = S.model_fn_wan_video_dual_stream(dit=pipe.dit, flow_stream=self.flow_stream, latents=rgb_latents,
                                                                       flow_latents=flow_latents, timestep=t, context=context,
                                                                       fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=False)
                rgb_latents = pipe.scheduler.step(rgb_pred, pipe.scheduler.timesteps[i], rgb_latents)
                flow_latents = pipe.scheduler.step(flow_pred, pipe.scheduler.timesteps[i], flow_latents)
                rgb_latents[:, :, :1], flow_latents[:, :, :1] = rgb_prefix, flow_prefix
        cond_ts = torch.full((1,), ACTION_COND_SIGMA, dtype=dtype, device=device)
        feats = S.capture_video_layer_features(dit=pipe.dit, flow_stream=self.flow_stream, latents=rgb_latents, flow_latents=flow_latents,
                                               timestep=cond_ts, context=context, fuse_vae_embedding_in_latents=True)
        layer_feats = [f.to(device=device, dtype=dtype) for f in S.concat_layer_feats(feats, 1)]
        n_rgb = rgb_latents.shape[2] * (rgb_latents.shape[3] // 2) * (rgb_latents.shape[4] // 2)
        per = (rgb_latents.shape[3] // 2) * (rgb_latents.shape[4] // 2)
        res = dict(layer_feats=layer_feats, action_context=action_context, rgb_latents=rgb_latents, flow_latents=flow_latents,
                   n_rgb=n_rgb, per_frame=per, T_lat=T_lat, grid=(T_lat, rgb_latents.shape[3] // 2, rgb_latents.shape[4] // 2))
        self.stage_cache[key] = res
        if len(self.stage_cache) > 3:
            self.stage_cache.popitem(last=False)
        return res

    def _segments(self, st: dict) -> Segments:
        n, per, T = st["n_rgb"], st["per_frame"], st["T_lat"]
        ranges = {"video_clean": (0, per), "video_future": (per, n), "flow_clean": (n, n + per), "flow_future": (n + per, 2 * n),
                  "track_cond": (2 * n, 2 * n), "track_anchor": (2 * n, 2 * n), "track_future": (2 * n, 2 * n), "action": (2 * n, 2 * n)}
        return Segments(ranges, st["grid"], (1, 0, 0))

    # -- intervention on the captured features ----------------------------------------------------
    def _apply(self, feats: list[torch.Tensor], iv: Intervention, seg: Segments) -> list[torch.Tensor]:
        if iv.mode == "none" or not iv.targets:
            return feats
        tgt = []
        for name in iv.targets:
            if name == "video_future":
                tgt += ["video_future", "flow_future"]
            elif name in seg.ranges and seg.length(name):
                tgt.append(name)
        K = seg.total
        keep_all = torch.ones(K, dtype=torch.bool)
        out = []
        for li, f in enumerate(feats):
            f = f.clone()
            keep = keep_all.clone()
            for name in tgt:
                a, b = seg.ranges[name]
                if iv.mode == "drop":
                    keep[a:b] = False
                elif iv.mode == "zero":
                    f[:, a:b] = 0
                elif iv.mode == "keep":
                    m = iv.keep["video_future"] if name in ("video_future", "flow_future") else iv.keep[name]
                    keep[a:b] = m.to(torch.bool)
                elif iv.mode == "pool":
                    f[:, a:b] = f[:, a:b].mean(dim=1, keepdim=True)
                elif iv.mode == "replace":
                    f[:, a:b] = iv.donor["feats"][li][:, a:b].to(f.device, f.dtype)
                else:
                    raise ValueError(iv.mode)
            out.append(f[:, keep.to(f.device)] if not keep.all() else f)
        return out

    # -- stage 3 ---------------------------------------------------------------------------------
    @torch.inference_mode()
    def predict_window(self, sample: dict, seed: int, *, intervention: Intervention | None = None, capture_attn: bool = False,
                       capture_kv: bool = False, static_future: bool = False, decode: bool = False) -> dict:
        st = self._stage12(sample, seed, static_future)
        seg = self._segments(st)
        self.segments = seg
        iv = intervention or Intervention.none()
        feats = self._apply(st["layer_feats"], iv, seg)
        ae, device, dtype = self.action_expert, self.device, self.dtype
        text_ctx = st["action_context"].to(device=device, dtype=dtype)
        sched = S.FlowMatchScheduler(num_inference_steps=ACTION_STEPS, shift=ACTION_SNR_SHIFT)
        g = torch.Generator(device=device).manual_seed(int(seed))
        lat = torch.randn(1, NUM_FRAMES, 14, generator=g, device=device, dtype=torch.float32).to(dtype)
        cur = torch.from_numpy(self.action_norm.normalize(np.asarray(sample["qpos0"], dtype=np.float32).reshape(1, -1)).astype(np.float32)).to(device=device, dtype=dtype)
        lat[:, 0] = cur
        cond_ts = torch.full((1,), ACTION_COND_SIGMA, device=device, dtype=dtype)
        attn_steps = []
        _PROBE.capture = capture_attn
        for i, timestep in enumerate(sched.timesteps):
            t = timestep.unsqueeze(0).to(dtype=dtype, device=device)
            sigma = sched.sigmas[i].to(device=device, dtype=dtype).reshape(1)
            _PROBE.reset()
            vel = ae(noisy_actions=lat, timestep=t.squeeze(), video_layer_feats=feats, text_context=text_ctx, joint_state=cur,
                     sigma=sigma, cond_timestep=cond_ts, return_x0=False)
            lat = sched.step(vel, sched.timesteps[i], lat)
            lat[:, 0] = cur
            if capture_attn and _PROBE.probs:
                layers = torch.stack(_PROBE.probs)                         # [L, B, h, A, K]
                attn_steps.append(dict(by_head=layers[:, 0].sum(dim=2).cpu(), by_query=layers[:, 0].mean(dim=1).cpu()))
        _PROBE.capture = False
        norm = lat[0].float().cpu().numpy()
        raw = self.action_norm.denormalize(norm).astype(np.float32)          # [33, 14] absolute qpos
        out = dict(action_norm=raw[1:], segments=seg, video_latents=st["rgb_latents"], track_latents=st["flow_latents"], normalized=norm[1:])
        if capture_attn:
            out["attn_by_head"] = torch.stack([s["by_head"] for s in attn_steps]).numpy()
            out["attn_by_query"] = torch.stack([s["by_query"] for s in attn_steps]).numpy()
        if capture_kv:
            out["kv"] = dict(feats=[f.detach().to("cpu") for f in st["layer_feats"]])
        if decode:
            self.pipe.load_models_to_device(["vae"])
            rgb = self.pipe.vae_output_to_video(self.pipe.vae.decode(st["rgb_latents"], device=device))
            flow = self.pipe.vae_output_to_video(self.pipe.vae.decode(st["flow_latents"], device=device))
            out["pred_rgb"] = np.stack([np.asarray(x) for x in rgb])
            out["pred_track"] = np.stack([np.asarray(x) for x in flow])
            self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        return out

    def action_errors(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        d = np.abs(pred - gt)
        out = dict(pos_cm=float("nan"), rot_deg=float("nan"), joint_deg=float(np.degrees(d[:, arm]).mean()),
                   left_joint_deg=float(np.degrees(d[:, :6]).mean()), right_joint_deg=float(np.degrees(d[:, 7:13]).mean()),
                   grip=float(0.5 * (d[:, 6].mean() + d[:, 13].mean())))
        z = (pred - gt) / (self.action_norm.std + 1e-6)
        out["norm_l2"] = float(np.sqrt((z ** 2).mean()))
        return out


__all__ = ["FlowWAMProbe"]
