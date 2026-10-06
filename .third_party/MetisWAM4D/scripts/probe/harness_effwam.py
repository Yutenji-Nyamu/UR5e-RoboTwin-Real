"""Probe harness for Efficient-WAM (official non-RT ``Efficient-WAM_stage3.pt``).

Efficient-WAM jointly denoises 3 latent frames of the 384x320 stacked image (condition frame + 2 future
frames = 360 tokens) and a 21-token action sequence (state + 16 actions + 4 registers) with fully
bidirectional attention in every one of the 12 compact-WAN layers, 10 steps for both branches.  The hook is
the module-level ``flash_attention`` used by ``WanSelfAttention`` (policy-local copy); the 21 action rows are
recomputed explicitly with the generic interventions, video rows are untouched.

Segments: ``video_clean`` [0,120), ``video_future`` [120,360), ``action`` [360,381); ``video_grid`` (3,12,10)
(head camera 240x320 on top -> token rows 0-7, wrists below).  Actions are absolute 14-D joint positions,
chunk 16 (compared with GT frames start+1..start+16); errors in joint degrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

EFF_ROOT = Path("/m2v_intern_v3/danglingwei/codes/tracking_wam_workspace/EfficientWAM_260817")
POLICY_DIR = EFF_ROOT / "inference" / "robotwin" / "EfficientWAM"
for p in (str(EFF_ROOT), str(POLICY_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import Intervention, ProbeAttention, Segments  # noqa: E402
from janusact4d_rt2imperfect_v1.data import ImperfectRoboTwinDataset, decode_source_jpeg  # noqa: E402
from openwam.dataloader.transforms.multiview import format_prompt_for_inference  # noqa: E402

from EfficientWAM.deploy_policy import build_runner  # noqa: E402
from EfficientWAM.paired_protocol import paired_stream_seed  # noqa: E402
from EfficientWAM.preprocess import preprocess_robotwin_observation  # noqa: E402
import EfficientWAM.third_party.wan.modules.model as wan_model_mod  # noqa: E402
from EfficientWAM.third_party.wan.modules.attention import flash_attention as _flash_attention  # noqa: E402

CAMERAS = ("head_camera", "left_camera", "right_camera")


class EffAttention(ProbeAttention):
    def __call__(self, q, k, v, **kwargs):
        if self.segments is None or k.shape[1] != self.segments.total or q.shape[1] != k.shape[1]:
            return _flash_attention(q, k, v, **kwargs)
        B, L, H, D = q.shape
        out = super().__call__(q.reshape(B, L, H * D), k.reshape(B, L, H * D), v.reshape(B, L, H * D), None, heads=H)
        return out.reshape(B, L, H, D).to(q.dtype)


class _WindowDataset:
    def __init__(self, base: ImperfectRoboTwinDataset, chunk: int):
        self.base, self.rows, self.instructions, self.chunk = base, base.rows, base.instructions, chunk

    def read_window(self, row_index: int, start: int) -> dict:
        import random
        row = self.rows[int(row_index)]
        with h5py.File(row["source"], "r") as src:
            total = int(src["observation/head_camera/depth"].shape[0])
            if start + 32 >= total:
                raise ValueError("window exceeds episode")
            # note: the stride-4 GT chunk extends to start+64 and is clipped at the episode end
            obs = {"observation": {c: {"rgb": np.asarray(decode_source_jpeg(src[f"observation/{c}/rgb"][start]))} for c in CAMERAS},
                   "joint_action": {"vector": src["joint_action/vector"][start].astype(np.float32)}}
            # Efficient-WAM predicts one action per 4 raw frames (10 Hz chunk over a 40 Hz recording); verified on
            # 36 windows: joint error 2.8 deg at stride 4 vs 7.0 deg at stride 1.
            T_ = src["joint_action/vector"].shape[0]
            idx = np.clip(start + 4 * np.arange(1, self.chunk + 1), 0, T_ - 1)
            gt = src["joint_action/vector"][:][idx].astype(np.float32)
        pool = self.instructions[int(row_index)].get("seen") or self.instructions[int(row_index)].get("unseen")
        instruction = random.choice(pool)
        return dict(prompt=format_prompt_for_inference(instruction), instruction=instruction, obs=obs, action=torch.from_numpy(gt),
                    key=f"{row['task']}/{row['variant']}/episode{row['episode']}/f{start}", bucket=row["bucket"],
                    head_rgb=torch.from_numpy(obs["observation"]["head_camera"]["rgb"].copy()))


class EffWAMProbe:
    def __init__(self, config_path: str | Path = POLICY_DIR / "deploy_policy.yml"):
        import yaml
        from harness import JANUS_ROOT
        self.device = torch.device("cuda:0")
        print("Building Efficient-WAM (stage3, non-RT)", flush=True)
        self.runner = build_runner(str(config_path), device="cuda")
        rt = self.runner.runtime
        self.steps = rt.num_inference_steps
        self.chunk = rt.chunk_size
        heads = self.runner.model.compact_wan.video_model.wan_model.blocks[0].self_attn.num_heads
        self.attn = EffAttention(heads)
        wan_model_mod.flash_attention = self.attn
        janus_cfg = yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
        self.dataset = _WindowDataset(ImperfectRoboTwinDataset(janus_cfg["data_root"], Path(janus_cfg["alpha_checkpoint"]) / "normalization_stats.npy"), self.chunk)
        self.segments: Segments | None = None
        self._std = None
        print("MODEL_READY", flush=True)

    def _measure_segments(self, video_latent, n_action_tokens: int) -> Segments:
        _, _, T, H, W = video_latent.shape
        per = (H // 2) * (W // 2)
        v_all = T * per
        ranges = {"video_clean": (0, per), "video_future": (per, v_all), "track_cond": (v_all, v_all), "track_anchor": (v_all, v_all),
                  "track_future": (v_all, v_all), "action": (v_all, v_all + n_action_tokens)}
        return Segments(ranges, (T, H // 2, W // 2), (1, 0, 0))

    @torch.inference_mode()
    def predict_window(self, sample: dict, seed: int, *, intervention: Intervention | None = None, capture_attn: bool = False,
                       capture_kv: bool = False, static_future: bool = False, decode: bool = False) -> dict:
        r, model, rt = self.runner, self.runner.model, self.runner.runtime
        processed = preprocess_robotwin_observation(sample["obs"], target_size=rt.video_size)
        r.set_instruction(sample["instruction"])
        video_dtype = model.compact_wan.video_model.precision
        action_dtype = next(model.action_expert.parameters()).dtype
        first_frame = processed["first_frame"].to(self.device, dtype=video_dtype)
        state = rt.action_normalizer.normalize(processed["state"].to(self.device, dtype=action_dtype))
        text_embeddings, _ = r._get_text_embeddings()

        def gen(stream):
            g = torch.Generator(device=self.device)
            g.manual_seed(paired_stream_seed(seed, stream))
            return g
        video_latent, condition_latent = r._initialize_video_latent(first_frame, gen("video"))
        noisy_actions = torch.randn((1, rt.chunk_size, model.config.action_dim), device=self.device, dtype=action_dtype, generator=gen("action"))
        r.action_scheduler.set_timesteps(num_inference_steps=rt.num_inference_steps, training=False)
        r.video_scheduler.set_timesteps(num_inference_steps=rt.num_video_inference_steps, training=False)
        action_ts = r.action_scheduler.timesteps.to(device=self.device, dtype=action_dtype)
        video_ts = r.video_scheduler.timesteps.to(device=self.device, dtype=video_dtype)
        n_action_tokens = 1 + rt.chunk_size + int(getattr(getattr(model.action_expert, "config", None), "num_registers", 4))
        self.segments = self.attn.segments = self._measure_segments(video_latent, n_action_tokens)
        self.attn.action_query_slice = self.segments.sl("action")
        self.attn.intervention = intervention or Intervention.none()
        self.attn.capture_attn, self.attn.capture_kv = capture_attn, capture_kv
        self.attn.kv_store = {}
        attn_steps = []
        for step in range(rt.num_inference_steps):
            if static_future:
                video_latent = condition_latent.expand(-1, -1, video_latent.shape[2], -1, -1).clone()
            batch = {"video_t": video_ts[step].expand(1), "initial_state": state, "noisy_actions": noisy_actions,
                     "action_t": action_ts[step].expand(1), "text_embeddings": text_embeddings, "video_latent": video_latent}
            self.attn.reset_forward(step)
            out = model(batch)
            if self.attn.segments.length("action") != n_action_tokens:
                raise RuntimeError("action token count mismatch")
            video_latent = r.video_scheduler.step(out["video_pred"], video_ts[step].expand(1), video_latent)
            video_latent[:, :, 0:1] = condition_latent
            noisy_actions = r.action_scheduler.step(out["action_pred"], action_ts[step].expand(1), noisy_actions)
            if capture_attn and self.attn.attn_layers:
                layers = torch.stack(self.attn.attn_layers)
                attn_steps.append(dict(by_head=layers[:, 0].sum(dim=2).cpu(), by_query=layers[:, 0].mean(dim=1).cpu()))
        actions = rt.action_normalizer.denormalize(noisy_actions)[0].float().cpu().numpy()    # [16, 14]
        out = dict(action_norm=actions, segments=self.segments, video_latents=video_latent, track_latents=None)
        if capture_attn:
            out["attn_by_head"] = torch.stack([s["by_head"] for s in attn_steps]).numpy()
            out["attn_by_query"] = torch.stack([s["by_query"] for s in attn_steps]).numpy()
        if capture_kv:
            out["kv"] = self.attn.kv_store
        if decode:
            with torch.inference_mode():
                pixels = model.compact_wan.decode_video(video_latent)                        # [1, 3, T, H, W] in [-1, 1]
            out["pred_rgb"] = ((pixels[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu().numpy()
            out["pred_track"] = np.zeros((0,), dtype=np.uint8)
        return out

    def action_errors(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        d = np.abs(pred - gt)
        if self._std is None:
            std = getattr(self.runner.runtime.action_normalizer, "_std", None)
            self._std = np.asarray(std.numpy() if std is not None else np.ones(14), dtype=np.float32).reshape(-1)
        z = (pred - gt) / (self._std + 1e-6)
        return dict(pos_cm=float("nan"), rot_deg=float("nan"), joint_deg=float(np.degrees(d[:, arm]).mean()),
                    left_joint_deg=float(np.degrees(d[:, :6]).mean()), right_joint_deg=float(np.degrees(d[:, 7:13]).mean()),
                    grip=float(0.5 * (d[:, 6].mean() + d[:, 13].mean())), norm_l2=float(np.sqrt((z ** 2).mean())))


__all__ = ["EffWAMProbe"]
