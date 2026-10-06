"""Inference-only probe harness for the JanusAct4D-RT2 three-expert baseline.

Run with ``/usr/bin/python`` and ``PYTHONPATH=$P/janusact4d_rt2imperfect_v1/vendor:$P`` where
``P=/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824`` (see ``scripts/probe/env.sh``).

What it adds on top of ``janusact4d_rt2imperfect_v1.evaluation.policy.Policy``:

* ``predict_window`` runs the exact closed-loop denoising recipe on a *dataset* window
  (``ImperfectRoboTwinDataset.read_window``) instead of a simulator observation;
* ``ProbeAttention`` replaces ``joint_attention`` so that, for the ACTION queries only, the
  future-Video / future-Track keys can be dropped, zeroed, pooled, sub-selected or replaced
  by K/V captured from a donor run (same denoising step and layer), while Video/Track rows
  keep the original fused attention -> world prediction is untouched;
* explicit softmax for the Action rows, so attention probabilities can be exported.

Key/value order per macro layer (verified against ``model.py``): ``[V_clean | V_future |
T_cond | T_anchor | T_future | A]``.  Segment boundaries are measured at run time.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import janusact4d_rt2imperfect_v1.model as janus_model
from janusact4d_rt2imperfect_v1.data import ImperfectRoboTwinDataset
from janusact4d_rt2imperfect_v1.evaluation.policy import Policy, eef20_to_sim
from openwam.deploy.denoise_schedule import schedule_sync
from training.observation import center_pad_video, prepare_condition_pixels

JANUS_ROOT = Path("/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824/janusact4d_rt2imperfect_v1")
JANUS_CKPT = Path("/ytech_milm_intern/danglingwei/outputs/JanusAct4D_imperfect/rt2_v1/simeval_19045/checkpoint/mp_rank_00_model_states.pt")
SEGMENT_NAMES = ("video_clean", "video_future", "track_cond", "track_anchor", "track_future", "action")
WORLD_SEGMENTS = ("video_future", "track_future")


@dataclass
class Segments:
    """Half-open key index ranges plus grid metadata."""
    ranges: dict[str, tuple[int, int]]
    video_grid: tuple[int, int, int]   # (latent frames, h, w) for video tokens (patch 2 already applied)
    track_grid: tuple[int, int, int]

    def sl(self, name: str) -> slice:
        a, b = self.ranges[name]
        return slice(a, b)

    def length(self, name: str) -> int:
        a, b = self.ranges[name]
        return b - a

    @property
    def total(self) -> int:
        return max(b for _, b in self.ranges.values())


@dataclass
class Intervention:
    """Applied to the keys the Action queries see, in ``targets`` segments only.

    mode:
      ``drop``     mask the target keys out (softmax renormalises over the rest)
      ``zero``     zero their K and V
      ``pool``     replace every target key/value by the mean over the kept target tokens
      ``keep``     keep only ``keep[seg]`` (bool over the segment), drop the rest
      ``replace``  use ``donor[(step, layer)][seg] = (k, v)`` captured from another run
    """
    mode: str = "none"
    targets: tuple[str, ...] = WORLD_SEGMENTS
    keep: dict[str, torch.Tensor] = field(default_factory=dict)
    donor: dict | None = None
    pool_per_frame: bool = False

    @staticmethod
    def none() -> "Intervention":
        return Intervention("none", ())


class ProbeAttention:
    """Drop-in for ``joint_attention``; owns the per-forward layer counter."""

    def __init__(self, heads: int):
        self.heads = heads
        self.segments: Segments | None = None
        self.intervention = Intervention.none()
        self.step = 0
        self.layer = -1
        self.capture_attn = False
        self.capture_kv = False
        self.attn_layers: list[torch.Tensor] = []      # per layer: [B, heads, A, K] float32 probabilities
        self.kv_store: dict = {}                       # (step, layer) -> {seg: (k, v)} (cpu, bf16)
        self.action_query_slice: slice | None = None   # rows of the action queries; None = the last A rows

    def reset_forward(self, step: int) -> None:
        self.step, self.layer = step, -1
        self.attn_layers = []

    # -- helpers -------------------------------------------------------------------------
    def _target_indices(self, name: str, device: torch.device) -> torch.Tensor:
        a, b = self.segments.ranges[name]
        return torch.arange(a, b, device=device)

    def _modify(self, k: torch.Tensor, v: torch.Tensor, kmask: torch.Tensor):
        iv = self.intervention
        if iv.mode == "none" or not iv.targets:
            return k, v, kmask
        k, v, kmask = k.clone(), v.clone(), kmask.clone()
        for name in iv.targets:
            idx = self._target_indices(name, k.device)
            if iv.mode == "drop":
                kmask[idx] = False
            elif iv.mode == "zero":
                k[:, idx] = 0
                v[:, idx] = 0
            elif iv.mode == "keep":
                keep = iv.keep[name].to(k.device)
                kmask[idx[~keep]] = False
            elif iv.mode == "pool":
                keep = iv.keep.get(name)
                sel = idx if keep is None else idx[keep.to(k.device)]
                if iv.pool_per_frame:
                    frames = self.segments.video_grid[0] - 1 if name == "video_future" else self.segments.track_grid[0] - 1
                    per = len(idx) // frames
                    for f in range(frames):
                        fi = idx[f * per:(f + 1) * per]
                        k[:, fi] = k[:, fi].mean(dim=1, keepdim=True)
                        v[:, fi] = v[:, fi].mean(dim=1, keepdim=True)
                else:
                    k[:, idx] = k[:, sel].mean(dim=1, keepdim=True)
                    v[:, idx] = v[:, sel].mean(dim=1, keepdim=True)
            elif iv.mode == "replace":
                dk, dv = iv.donor[(self.step, self.layer)][name]
                k[:, idx] = dk.to(k.device, k.dtype)
                v[:, idx] = dv.to(v.device, v.dtype)
            else:
                raise ValueError(iv.mode)
        return k, v, kmask

    # -- the attention --------------------------------------------------------------------
    def __call__(self, query, key, value, mask, *, heads):
        """``query/key/value: [B, L, H*D]``; ``mask: [Q, K]`` bool (True = visible) or None."""
        self.layer += 1
        seg = self.segments
        B, Q, width = query.shape
        K = key.shape[1]
        A = seg.length("action")
        d = width // heads
        if K != seg.total:
            raise RuntimeError(f"key length {K} != measured segments {seg.ranges}")
        rows = self.action_query_slice or slice(Q - A, Q)

        def split(x):
            return x.reshape(B, x.shape[1], heads, d).transpose(1, 2)

        if self.capture_kv:
            entry = {}
            for name in ("video_clean",) + WORLD_SEGMENTS:
                s = seg.sl(name)
                entry[name] = (key[:, s].detach().to("cpu", copy=True), value[:, s].detach().to("cpu", copy=True))
            self.kv_store[(self.step, self.layer)] = entry

        # every row with the original fused attention, then the action rows are recomputed explicitly
        out = F.scaled_dot_product_attention(split(query), split(key), split(value), attn_mask=mask)   # [B, h, Q, d]
        kmask = mask[rows][0] if mask is not None else torch.ones(K, dtype=torch.bool, device=key.device)
        k_mod, v_mod, kmask = self._modify(key, value, kmask)
        q_a = split(query[:, rows])                                    # [B, h, A, d]
        logits = torch.matmul(q_a, split(k_mod).transpose(-1, -2)) / math.sqrt(d)   # [B, h, A, K]
        logits = logits.float().masked_fill(~kmask[None, None, None, :], float("-inf"))
        probs = logits.softmax(dim=-1)
        if self.capture_attn:
            if getattr(self, "keep_grad", False) and probs.requires_grad:
                probs.retain_grad()
                self.attn_layers.append(probs)
            else:
                self.attn_layers.append(probs.detach())
        out_a = torch.matmul(probs.to(v_mod.dtype), split(v_mod))    # [B, h, A, d]
        out = out.clone()
        out[:, :, rows] = out_a
        return out.transpose(1, 2).reshape(B, Q, width)


class JanusProbe(Policy):
    """``Policy`` + dataset-window inference + attention interventions."""

    def __init__(self, config: dict | None = None, checkpoint: str | Path = JANUS_CKPT, denoise_steps: int = 10):
        import yaml
        config = config or yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
        super().__init__(config, str(checkpoint), denoise_steps)
        self.config = config
        self.attn = ProbeAttention(self.alpha.video_backbone.num_heads)
        janus_model.joint_attention = self.attn        # model.py imported the name -> patch there
        self.dataset = ImperfectRoboTwinDataset(config["data_root"], Path(config["alpha_checkpoint"]) / "normalization_stats.npy")
        self.segments: Segments | None = None

    # -- segments ----------------------------------------------------------------------------
    def _measure_segments(self, inputs, track, encoded) -> Segments:
        vb = self.alpha.video_backbone
        lat = inputs["latents"]                                    # [1, 48, F, H, W]
        f, h, w = lat.shape[2], lat.shape[3] // 2, lat.shape[4] // 2
        v_clean, v_all = h * w, f * h * w
        tf, th, tw = track.shape[2], track.shape[3] // 2, track.shape[4] // 2
        t_cond = th * tw                                           # fused rgb/depth/mask condition grid
        t_anchor = th * tw
        t_all = t_cond + t_anchor + (tf - 1) * th * tw
        ranges = {
            "video_clean": (0, v_clean), "video_future": (v_clean, v_all),
            "track_cond": (v_all, v_all + t_cond), "track_anchor": (v_all + t_cond, v_all + t_cond + t_anchor),
            "track_future": (v_all + t_cond + t_anchor, v_all + t_all), "action": (v_all + t_all, v_all + t_all + 32),
        }
        return Segments(ranges, (f, h, w), (tf, th, tw))

    # -- inference on a dataset window --------------------------------------------------------
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
        rgb = sample["head_rgb"][None]
        depth = sample["head_depth"][None].float()
        mask_tensor = sample["head_mask"][None]
        conditions = prepare_condition_pixels(rgb, depth, mask_tensor, target_height=256, target_width=320,
                                              depth_scale=.001, depth_min_m=self.encoder.depth_min_m,
                                              depth_max_m=self.encoder.depth_max_m)
        encoded = self.encoder.encode_pixels(torch.cat(conditions, dim=0))
        anchor_pixels = mask_tensor[..., None].expand(-1, -1, -1, 3).to(torch.uint8) * 128
        anchor = self.encoder.encode_pixels(center_pad_video(anchor_pixels[:, None], 256, 320))
        track = torch.randn((1, anchor.shape[1], 3, *anchor.shape[-2:]), device=self.device, dtype=self.dtype)
        track[:, :, :1] = anchor
        action_noise = torch.randn((1, 32, 80), device=self.device, dtype=self.dtype)
        action = action_noise.clone()
        active = sample["proprio_mask"][0].to(self.device)
        proprio = sample["proprio"].to(self.device, self.dtype)[None]

        self.segments = self.attn.segments = self._measure_segments(inputs, track, encoded)
        self.attn.intervention = intervention or Intervention.none()
        self.attn.capture_attn, self.attn.capture_kv = capture_attn, capture_kv
        self.attn.kv_store = {}
        attn_steps = []
        schedule = schedule_sync(vb.scheduler, ab.scheduler, self.steps, shift=ab.shift_action, shift_video=vb.shift_video)
        for step, ((tv, ta), (tv_next, ta_next)) in enumerate(zip(schedule[:-1], schedule[1:])):
            sv, sa = tv / vb.scheduler.num_train_timesteps, ta / ab.scheduler.num_train_timesteps
            svn, san = tv_next / vb.scheduler.num_train_timesteps, ta_next / ab.scheduler.num_train_timesteps
            inputs["timestep"] = torch.tensor([tv], device=self.device, dtype=self.dtype)
            if static_future:
                inputs["latents"][:, :, 1:] = ref
            self.attn.reset_forward(step)
            pred = self.model(noisy_actions=action,
                              action_timestep=torch.tensor([ta], device=self.device, dtype=self.dtype),
                              noisy_track=track, track_timestep=torch.tensor([sa * 1000], device=self.device),
                              rgb_condition=encoded[:1], depth_condition=encoded[1:2], mask_condition=encoded[2:],
                              proprio=proprio, proprio_mask=active[None, None], pipeline_inputs=inputs)
            inputs["latents"] = inputs["latents"] + pred.video * (svn - sv)
            inputs["latents"][:, :, :1] = ref
            track = track + pred.track * (san - sa)
            track[:, :, :1] = anchor
            action = ab.scheduler.flow_step(pred.action, sa, san, action)
            action[..., ~active] = action_noise[..., ~active] * san
            if capture_attn:
                # aggregate on the fly: head-resolved sum over action queries, and head-mean per query
                layers = torch.stack(self.attn.attn_layers)              # [L, B, h, A, K]
                attn_steps.append(dict(
                    by_head=layers[:, 0].sum(dim=2).cpu(),               # [L, h, K]
                    by_query=layers[:, 0].mean(dim=1).cpu(),             # [L, A, K]
                ))
        values = action[0].float().cpu().numpy()
        out = dict(action_norm=values, segments=self.segments, video_latents=inputs["latents"], track_latents=track)
        if capture_attn:
            out["attn_by_head"] = torch.stack([s["by_head"] for s in attn_steps]).numpy()    # [S, L, h, K]
            out["attn_by_query"] = torch.stack([s["by_query"] for s in attn_steps]).numpy()  # [S, L, A, K]
        if capture_kv:
            out["kv"] = self.attn.kv_store
        if decode:
            out["pred_rgb"] = np.stack([np.asarray(x) for x in vb.decode_video(inputs["latents"], tiled=False)])
            out["pred_track"] = ((self.encoder.decode(track)[0].float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
        return out

    # -- metrics -----------------------------------------------------------------------------
    def action_errors(self, pred_norm: np.ndarray, gt_norm: np.ndarray) -> dict:
        """Native EEF errors between two normalised (32, 80) action blocks: cm / deg / gripper."""
        def native(a):
            return self.normalizer.unnormalize(np.concatenate((a[:, :10], a[:, 34:44]), axis=-1))
        p, g = native(pred_norm), native(gt_norm)
        ps, gs = eef20_to_sim(p), eef20_to_sim(g)
        out = {}
        for i, arm in enumerate(("left", "right")):
            o = i * 7
            pos = np.linalg.norm(ps[:, o:o + 3] - gs[:, o:o + 3], axis=-1) * 100
            qa, qb = ps[:, o + 3:o + 7], gs[:, o + 3:o + 7]
            dot = np.clip(np.abs((qa * qb).sum(-1)) / (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1) + 1e-8), 0, 1)
            rot = np.degrees(2 * np.arccos(dot))
            grip = np.abs(ps[:, o + 6] - gs[:, o + 6])
            out[f"{arm}_pos_cm"] = float(pos.mean())
            out[f"{arm}_rot_deg"] = float(rot.mean())
            out[f"{arm}_grip"] = float(grip.mean())
        out["pos_cm"] = 0.5 * (out["left_pos_cm"] + out["right_pos_cm"])
        out["rot_deg"] = 0.5 * (out["left_rot_deg"] + out["right_rot_deg"])
        out["grip"] = 0.5 * (out["left_grip"] + out["right_grip"])
        out["norm_l2"] = float(np.sqrt(((pred_norm - gt_norm)[:, list(range(10)) + list(range(34, 44))] ** 2).mean()))
        return out


def load_windows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


__all__ = ["Intervention", "JanusProbe", "ProbeAttention", "Segments", "WORLD_SEGMENTS", "SEGMENT_NAMES", "load_windows"]
