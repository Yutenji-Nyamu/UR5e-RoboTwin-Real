"""Probe harness for FastWAM (official ``robotwin_uncond_3cam_384.pt``).

FastWAM's deployed path (``infer_action``) never generates a future: the video expert is run once on the
current frame, its per-layer K/V (120 tokens) are cached, and the 32 action tokens attend to
``[cached current-frame video K/V | action K/V]`` for 10 denoising steps.  So the only world information
the action can read is the *observed* frame; the future is structurally absent.  Hook:
``MoT._mixed_attention`` (``[B, S, H*D]`` tensors + 2-D bool mask, same calling convention as OpenWAM-Alpha).
Use ``--pass p31c`` (current-frame interventions); ``p31`` conditions on ``video_future`` are no-ops here.

Segments: ``video_clean`` [0,120), ``video_future`` empty, ``action`` [120,152).  Actions: 14-D joint qpos,
horizon 32; errors in joint degrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

FAST_ROOT = Path("/m2v_intern_v3/danglingwei/codes/FastWAM_260625")
CKPT = Path("/m2v_intern/_public_models/yuanty/fastwam/robotwin_uncond_3cam_384.pt")
STATS = CKPT.with_name("robotwin_uncond_3cam_384_dataset_stats.json")
for p in (str(FAST_ROOT / "src"), str(FAST_ROOT / "experiments" / "robotwin")):
    if p not in sys.path:
        sys.path.insert(0, p)
import os  # noqa: E402

os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/m2v_intern/_public_models")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

from harness import Intervention, ProbeAttention, Segments  # noqa: E402
from janusact4d_rt2imperfect_v1.data import ImperfectRoboTwinDataset, decode_source_jpeg  # noqa: E402
from openwam.dataloader.transforms.multiview import format_prompt_for_inference  # noqa: E402

from fastwam_policy.deploy_policy import DEFAULT_PROMPT, get_model  # noqa: E402

CAMERAS = ("head_camera", "left_camera", "right_camera")


class FastAttention(ProbeAttention):
    def __init__(self, heads: int, original):
        super().__init__(heads)
        self.original = original

    def __call__(self, q_cat, k_cat, v_cat, attention_mask):
        if self.segments is None or k_cat.shape[1] != self.segments.total or q_cat.shape[1] != self.segments.length("action"):
            return self.original(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)
        return super().__call__(q_cat, k_cat, v_cat, attention_mask.to(q_cat.device), heads=self.heads)


class _WindowDataset:
    def __init__(self, base: ImperfectRoboTwinDataset, horizon: int):
        self.base, self.rows, self.instructions, self.horizon = base, base.rows, base.instructions, horizon

    def read_window(self, row_index: int, start: int) -> dict:
        import random
        row = self.rows[int(row_index)]
        with h5py.File(row["source"], "r") as src:
            total = int(src["observation/head_camera/depth"].shape[0])
            if start + 32 >= total:
                raise ValueError("window exceeds episode")
            obs = {"observation": {c: {"rgb": np.asarray(decode_source_jpeg(src[f"observation/{c}/rgb"][start]))} for c in CAMERAS},
                   "joint_action": {"vector": src["joint_action/vector"][start].astype(np.float32)}}
            gt = src["joint_action/vector"][start + 1:start + 1 + self.horizon].astype(np.float32)
        pool = self.instructions[int(row_index)].get("seen") or self.instructions[int(row_index)].get("unseen")
        instruction = random.choice(pool)
        return dict(prompt=format_prompt_for_inference(instruction), instruction=instruction, obs=obs, action=torch.from_numpy(gt),
                    key=f"{row['task']}/{row['variant']}/episode{row['episode']}/f{start}", bucket=row["bucket"],
                    head_rgb=torch.from_numpy(obs["observation"]["head_camera"]["rgb"].copy()))


class FastWAMProbe:
    def __init__(self, num_inference_steps: int = 10):
        import yaml
        from harness import JANUS_ROOT
        self.device = torch.device("cuda:0")
        print("Building FastWAM (robotwin_uncond_3cam_384)", flush=True)
        self.policy = get_model(dict(ckpt_setting=str(CKPT), dataset_stats_path=str(STATS), sim_cfg_name="sim_robotwin.yaml",
                                     sim_task="robotwin_uncond_3cam_384_1e-4", num_inference_steps=num_inference_steps, seed=0,
                                     device="cuda", mixed_precision="bf16"))
        self.model = self.policy.model
        self.steps = num_inference_steps
        mot = self.model.mot
        self.attn = FastAttention(mot.num_heads, mot._mixed_attention)
        mot._mixed_attention = self.attn
        janus_cfg = yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
        self.dataset = _WindowDataset(ImperfectRoboTwinDataset(janus_cfg["data_root"], Path(janus_cfg["alpha_checkpoint"]) / "normalization_stats.npy"),
                                      int(self.policy.action_horizon))
        self.segments: Segments | None = None
        self._std = None
        print("MODEL_READY", flush=True)

    @torch.inference_mode()
    def predict_window(self, sample: dict, seed: int, *, intervention: Intervention | None = None, capture_attn: bool = False,
                       capture_kv: bool = False, static_future: bool = False, decode: bool = False) -> dict:
        pol = self.policy
        image = pol._build_robotwin_image_tensor(sample["obs"])
        proprio = pol._normalize_state(np.asarray(sample["obs"]["joint_action"]["vector"], dtype=np.float32))
        # segments: 384x320 -> 24x20 latent -> patch 2 -> 12x10 = 120 tokens for the single (clean) frame
        per = (image.shape[-2] // 16 // 2) * (image.shape[-1] // 16 // 2)
        A = int(pol.action_horizon)
        self.segments = self.attn.segments = Segments({"video_clean": (0, per), "video_future": (per, per), "track_cond": (per, per),
                                                       "track_anchor": (per, per), "track_future": (per, per), "action": (per, per + A)},
                                                      (1, image.shape[-2] // 32, image.shape[-1] // 32), (1, 0, 0))
        self.attn.action_query_slice = slice(0, A)             # action-only queries in the cached path
        self.attn.intervention = intervention or Intervention.none()
        self.attn.capture_attn, self.attn.capture_kv = capture_attn, capture_kv
        self.attn.kv_store = {}
        # ``infer_action`` runs the whole 10-step loop internally; the hook counts layers per forward, so we
        # detect step boundaries by the layer counter wrapping around.
        attn_steps: list = []
        n_layers = self.model.mot.num_layers
        probe = self.attn
        orig_call = FastAttention.__call__

        def counting_call(q_cat, k_cat, v_cat, attention_mask):
            q, k, v, m = q_cat, k_cat, v_cat, attention_mask
            if probe.segments is not None and k.shape[1] == probe.segments.total and q.shape[1] == A:
                if probe.layer + 1 >= n_layers:
                    if capture_attn and probe.attn_layers:
                        layers = torch.stack(probe.attn_layers)
                        attn_steps.append(dict(by_head=layers[:, 0].sum(dim=2).cpu(), by_query=layers[:, 0].mean(dim=1).cpu()))
                    probe.reset_forward(probe.step + 1)
            return orig_call(probe, q, k, v, m)

        probe.reset_forward(0)
        self.model.mot._mixed_attention = counting_call
        try:
            pred = self.model.infer_action(prompt=DEFAULT_PROMPT.format(task=sample["instruction"]), input_image=image, action_horizon=A, proprio=proprio, negative_prompt=pol.negative_prompt,
                                           text_cfg_scale=pol.text_cfg_scale, num_inference_steps=self.steps, sigma_shift=pol.sigma_shift,
                                           seed=int(seed), rand_device=pol.rand_device, tiled=pol.tiled)
        finally:
            self.model.mot._mixed_attention = self.attn
        if capture_attn and probe.attn_layers:
            layers = torch.stack(probe.attn_layers)
            attn_steps.append(dict(by_head=layers[:, 0].sum(dim=2).cpu(), by_query=layers[:, 0].mean(dim=1).cpu()))
        actions = pol._denormalize_action(pred["action"])[0]          # [32, 14]
        out = dict(action_norm=actions.astype(np.float32), segments=self.segments, video_latents=None, track_latents=None)
        if capture_attn:
            out["attn_by_head"] = torch.stack([s["by_head"] for s in attn_steps]).numpy()
            out["attn_by_query"] = torch.stack([s["by_query"] for s in attn_steps]).numpy()
        if capture_kv:
            out["kv"] = self.attn.kv_store
        if decode:
            out["pred_rgb"] = np.zeros((0,), dtype=np.uint8)
            out["pred_track"] = np.zeros((0,), dtype=np.uint8)
        return out

    def action_errors(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        d = np.abs(pred - gt)
        return dict(pos_cm=float("nan"), rot_deg=float("nan"), joint_deg=float(np.degrees(d[:, arm]).mean()),
                    left_joint_deg=float(np.degrees(d[:, :6]).mean()), right_joint_deg=float(np.degrees(d[:, 7:13]).mean()),
                    grip=float(0.5 * (d[:, 6].mean() + d[:, 13].mean())), norm_l2=float(np.sqrt(((pred - gt) ** 2).mean())))


__all__ = ["FastWAMProbe"]
