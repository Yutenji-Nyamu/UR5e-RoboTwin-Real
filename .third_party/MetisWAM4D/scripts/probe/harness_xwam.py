"""Probe harness for X-WAM (official ``robotwin_sft`` checkpoint).

X-WAM denoises one joint sequence ``[video (3 latent frames x 3 views x 80) | 32 actions | 9 proprio]``
through the Wan2.2 DiT; every block's ``WanSelfAttention`` is the place where the action tokens read the
(partially denoised) future video.  We hook ``modules.wan_model.attention`` for the self-attention calls
(the text cross-attention calls are recognised by their key length and passed through) and recompute the
32 action rows explicitly with the generic :class:`harness.ProbeAttention` interventions.

Deployed protocol (``LocalXWAMPolicy``): 50-step video schedule, actions finished after 10 joint steps
(``early_stop``), depth branch off (``run_depth=False``), cfg 0.  ``predict_window`` reproduces it; with
``decode=True`` the remaining 40 video-only steps are run to obtain the predicted RGB.

Segments: ``video_clean`` = latent frame 0 (current frame, all views), ``video_future`` = latent frames 1-2,
``action`` = 32 tokens, ``proprio`` = 9 tokens (kept, never intervened).  ``video_grid`` is reported as
``(3, 24, 10)``: the three 8x10 views stacked (rows 0-7 head, 8-15 left wrist, 16-23 right wrist), which is
what ``analyze_openloop`` expects for the head-view slice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

XWAM_ROOT = Path("/m2v_intern_v3/danglingwei/codes/tracking_wam_workspace/XWAM_260716")
XWAM_EXP = Path("/m2v_intern/_public_models/sharinka0715/X-WAM-checkpoints/robotwin_sft")
WAN_DIR = Path("/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B")
if str(XWAM_ROOT) not in sys.path:
    sys.path.insert(0, str(XWAM_ROOT))

from harness import Intervention, ProbeAttention, Segments  # noqa: E402
from janusact4d_rt2imperfect_v1.data import ImperfectRoboTwinDataset, decode_source_jpeg  # noqa: E402
from openwam.dataloader.transforms.multiview import format_prompt_for_inference  # noqa: E402

import modules.wan_model as wan_model  # noqa: E402
from custom_exps.simeval_inplace.local_policy import (  # noqa: E402
    BASE_COORD_XFORM, EEF_AXES_XFORM, LocalXWAMPolicy, PolicyConfig, compute_future_poses, resize_and_center_crop_tensor)
from modules.attention import attention as xwam_attention  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

PROMPT_PREFIX = format_prompt_for_inference("")
CAMERAS = ("head_camera", "left_camera", "right_camera")


class XWAMAttention(ProbeAttention):
    """Adapter for ``attention(q, k, v, ...)`` with ``[B, L, H, D]`` tensors and no mask."""

    def __call__(self, q, k, v, **kwargs):  # noqa: D401
        if self.segments is None or k.shape[1] != self.segments.total or q.shape[1] != k.shape[1]:
            return xwam_attention(q, k, v, **kwargs)                    # text cross-attention etc.
        B, L, H, D = q.shape
        out = super().__call__(q.reshape(B, L, H * D), k.reshape(B, L, H * D), v.reshape(B, L, H * D), None, heads=H)
        return out.reshape(B, L, H, D).to(q.dtype)


class _WindowDataset:
    """Dataset-window reader in X-WAM's observation contract (shares rows/instructions with the RT2 loader)."""

    def __init__(self, base: ImperfectRoboTwinDataset):
        self.base = base
        self.rows = base.rows
        self.instructions = base.instructions

    def read_window(self, row_index: int, start: int) -> dict:
        import random
        row = self.rows[int(row_index)]
        with h5py.File(row["source"], "r") as src:
            total = int(src["observation/head_camera/depth"].shape[0])
            if start + 32 >= total:
                raise ValueError("window exceeds episode")
            rgbs = np.stack([np.asarray(decode_source_jpeg(src[f"observation/{c}/rgb"][start])) for c in CAMERAS])   # [3, H, W, 3]
            frames = list(range(start, start + 33))
            ep = {}
            for side in ("left", "right"):
                ep[side] = np.concatenate([src[f"endpose/{side}_endpose"][frames], src[f"endpose/{side}_gripper"][frames][:, None]], axis=1)  # [33, 8]
        prompt_pool = self.instructions[int(row_index)].get("seen") or self.instructions[int(row_index)].get("unseen")
        instruction = random.choice(prompt_pool)
        obs = dict(observation={c: dict(rgb=rgbs[i]) for i, c in enumerate(CAMERAS)},
                   endpose=dict(left_endpose=ep["left"][0, :7], left_gripper=ep["left"][0, 7],
                                right_endpose=ep["right"][0, :7], right_gripper=ep["right"][0, 7]))
        gt16 = np.concatenate([ep["left"][1:], ep["right"][1:]], axis=1).astype(np.float32)      # [32, 16] xyz quat grip x2
        return dict(prompt=format_prompt_for_inference(instruction), instruction=instruction, obs=obs,
                    action=torch.from_numpy(gt16), key=f"{row['task']}/{row['variant']}/episode{row['episode']}/f{start}",
                    bucket=row["bucket"], head_rgb=torch.from_numpy(rgbs[0].copy()))


class XWAMProbe:
    def __init__(self, denoise_steps: int = 50, action_denoise_steps: int = 10):
        import yaml
        from harness import JANUS_ROOT
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        cfg = PolicyConfig(exp_path=str(XWAM_EXP), steps="last", wan_checkpoint_dir=str(WAN_DIR),
                           denoise_steps=denoise_steps, action_denoise_steps=action_denoise_steps, cfg=0.0,
                           compile_model=False, empty_cache_after_infer=False)
        print("Building X-WAM (robotwin_sft, last.ckpt)", flush=True)
        self.policy = LocalXWAMPolicy(cfg)
        self.runner = self.policy.model
        self.config = self.policy.config
        self.steps = action_denoise_steps
        self.attn = XWAMAttention(self.runner.model.blocks[0].self_attn.num_heads)
        wan_model.attention = self.attn                                  # module-level name used by WanSelfAttention
        janus_cfg = yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
        self.dataset = _WindowDataset(ImperfectRoboTwinDataset(janus_cfg["data_root"], Path(janus_cfg["alpha_checkpoint"]) / "normalization_stats.npy"))
        self.segments: Segments | None = None
        print("MODEL_READY", flush=True)

    # -- segments ----------------------------------------------------------------------------
    def _measure_segments(self, latents, Ta: int, Tp: int) -> Segments:
        B, C, T, V, H, W = latents.shape
        per_frame = V * (H // 2) * (W // 2)
        v_clean, v_all = per_frame, T * per_frame
        ranges = {"video_clean": (0, v_clean), "video_future": (v_clean, v_all),
                  "track_cond": (v_all, v_all), "track_anchor": (v_all, v_all), "track_future": (v_all, v_all),
                  "action": (v_all, v_all + Ta), "proprio": (v_all + Ta, v_all + Ta + Tp)}
        return Segments(ranges, (T, V * (H // 2), W // 2), (1, 0, 0))

    # -- inference ---------------------------------------------------------------------------
    @torch.inference_mode()
    def predict_window(self, sample: dict, seed: int, *, intervention: Intervention | None = None,
                       capture_attn: bool = False, capture_kv: bool = False, static_future: bool = False,
                       decode: bool = False) -> dict:
        from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore
        pol, runner, config = self.policy, self.runner, self.config
        rgbs, left_arm, right_arm, robot_states = pol._observation_to_proprio(sample["obs"])
        rgb = torch.from_numpy(rgbs).bfloat16().unsqueeze(0).cuda()
        rgb = rgb.permute(0, 1, 4, 2, 3)
        rgb = resize_and_center_crop_tensor(rgb, config.dataset.video_size, 0.95)
        proprio_norm = 2 * (robot_states - pol.state_q01) / (pol.state_q99 - pol.state_q01) - 1
        proprio = torch.from_numpy(proprio_norm).bfloat16().unsqueeze(0).cuda()
        T = config.frame_num
        Ta = (config.frame_num - 1) * runner.model.action_num
        Tp = config.frame_num
        batch = {"video": rgb.unsqueeze(2).repeat(1, 1, T, 1, 1, 1), "proprios": proprio.unsqueeze(1).repeat(1, Tp, 1),
                 "actions": torch.zeros((1, Ta, config.action_dim), device=self.device, dtype=proprio.dtype),
                 "prompt": [sample["instruction"]]}
        runner.run_depth = False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context, gt_latents, _ = runner._prepare_condition(batch)
        B, C, Tl, V, H, W = gt_latents.shape
        gt_actions, gt_proprios = batch["actions"].float(), batch["proprios"].float()
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        noise_latents = torch.randn(gt_latents.shape, generator=gen, device=self.device, dtype=gt_latents.dtype)
        noise_actions = torch.randn(gt_actions.shape, generator=gen, device=self.device, dtype=gt_actions.dtype)
        noise_proprios = torch.randn(gt_proprios.shape, generator=gen, device=self.device, dtype=gt_proprios.dtype)
        latent_mask = torch.zeros((B, 1, Tl, 1, 1, 1), dtype=torch.long, device=self.device)
        latent_mask[:, :, 0] = 1
        xt_latents = gt_latents * latent_mask + noise_latents * (1 - latent_mask)
        action_mask = torch.zeros((B, Ta, 1), dtype=torch.long, device=self.device)
        xt_actions = noise_actions.clone()
        proprio_mask = torch.zeros((B, Tp, 1), dtype=torch.long, device=self.device)
        proprio_mask[:, 0] = 1
        xt_proprios = gt_proprios * proprio_mask + noise_proprios * (1 - proprio_mask)

        def sched(n):
            s = FlowUniPCMultistepScheduler(num_train_timesteps=config.flow_matching_num_train_timesteps, shift=1, use_dynamic_shifting=False)
            s.set_timesteps(n, device=self.device, shift=config.time_shifting)
            return s
        s_video, s_act, s_prop = sched(config.sample_steps), sched(self.steps), sched(self.steps)
        self.segments = self.attn.segments = self._measure_segments(gt_latents, Ta, Tp)
        self.attn.action_query_slice = self.segments.sl("action")
        self.attn.intervention = intervention or Intervention.none()
        self.attn.capture_attn, self.attn.capture_kv = capture_attn, capture_kv
        self.attn.kv_store = {}
        attn_steps = []
        n_steps = config.sample_steps if decode else self.steps
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for ti in range(n_steps):
                video_t = s_video.timesteps[ti]
                joint = ti < self.steps
                action_t = s_act.timesteps[ti] if joint else 0
                proprio_t = s_prop.timesteps[ti] if joint else 0
                if static_future:
                    xt_latents = gt_latents.clone()                        # every latent frame = current frame
                latent_ts = video_t * (1 - latent_mask).view(B, Tl)
                action_ts = action_t * (1 - action_mask).view(B, Ta)
                proprio_ts = proprio_t * (1 - proprio_mask).view(B, Tp)
                self.attn.reset_forward(ti)
                self.attn.capture_attn = capture_attn and joint
                v_lat, v_act, v_prop, _ = runner.model(x=xt_latents, t=latent_ts, context=context, actions=xt_actions,
                                                       t_actions=action_ts, proprios=xt_proprios, t_proprios=proprio_ts,
                                                       cfg=0.0, run_depth=False)
                xt_latents = s_video.step(v_lat, video_t, xt_latents, return_dict=False)[0]
                xt_latents = gt_latents * latent_mask + xt_latents * (1 - latent_mask)
                if joint:
                    xt_actions = s_act.step(v_act, action_t, xt_actions, return_dict=False)[0]
                    xt_proprios = s_prop.step(v_prop, proprio_t, xt_proprios, return_dict=False)[0]
                    xt_proprios = gt_proprios * proprio_mask + xt_proprios * (1 - proprio_mask)
                    if capture_attn and self.attn.attn_layers:
                        layers = torch.stack(self.attn.attn_layers)
                        attn_steps.append(dict(by_head=layers[:, 0].sum(dim=2).cpu(), by_query=layers[:, 0].mean(dim=1).cpu()))
        acts = xt_actions[0].float().cpu().numpy()
        acts = (acts[:, : pol.action_dim] + 1) / 2 * (pol.action_q99 - pol.action_q01) + pol.action_q01
        poses16 = pol._deltas_to_ee_actions(left_arm, right_arm, acts).astype(np.float32)        # [32, 16]
        out = dict(action_norm=poses16, segments=self.segments, video_latents=xt_latents, track_latents=None, deltas=acts)
        if capture_attn:
            out["attn_by_head"] = torch.stack([s["by_head"] for s in attn_steps]).numpy()
            out["attn_by_query"] = torch.stack([s["by_query"] for s in attn_steps]).numpy()
        if capture_kv:
            out["kv"] = self.attn.kv_store
        if decode:
            from einops import rearrange
            lat = rearrange(xt_latents, "b c t v h w -> (b v) c t h w")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vid = runner.vae.decode(lat)                                # [(b v), 3, T, H, W]
            vid = torch.clamp((vid.float() + 1) * 127.5, 0, 255).byte()
            out["pred_rgb"] = rearrange(vid, "(b v) c t h w -> t h (b v w) c", v=V).cpu().numpy()   # [T, 256, 960, 3] head|left|right
            out["pred_track"] = np.zeros((0,), dtype=np.uint8)
        return out

    # -- metrics (absolute EEF space, same fields as the other probes) -----------------------------
    @staticmethod
    def action_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
        out = {}
        for i, arm in enumerate(("left", "right")):
            o = i * 8
            pos = np.linalg.norm(pred[:, o:o + 3] - gt[:, o:o + 3], axis=-1) * 100
            qa, qb = pred[:, o + 3:o + 7], gt[:, o + 3:o + 7]
            dot = np.clip(np.abs((qa * qb).sum(-1)) / (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1) + 1e-8), 0, 1)
            rot = np.degrees(2 * np.arccos(dot))
            grip = np.abs(pred[:, o + 7] - gt[:, o + 7])
            out[f"{arm}_pos_cm"], out[f"{arm}_rot_deg"], out[f"{arm}_grip"] = float(pos.mean()), float(rot.mean()), float(grip.mean())
        out["pos_cm"] = 0.5 * (out["left_pos_cm"] + out["right_pos_cm"])
        out["rot_deg"] = 0.5 * (out["left_rot_deg"] + out["right_rot_deg"])
        out["grip"] = 0.5 * (out["left_grip"] + out["right_grip"])
        out["norm_l2"] = float(np.sqrt(((pred - gt) ** 2).mean()))
        return out


__all__ = ["XWAMProbe", "XWAMAttention"]
