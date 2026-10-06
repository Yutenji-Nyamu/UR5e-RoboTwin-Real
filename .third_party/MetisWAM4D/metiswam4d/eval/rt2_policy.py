"""MetisWAM4D policy for RoboTwin 2.0: one training-domain observation -> 32 simulator EEF actions.

Every input goes through the functions of the RT2 training pipeline (``RT2EpisodeDataset`` /
``RT2OnlineEncoder``): the 3-view L layout, the head-camera RGB / depth / mask condition images,
the zero-displacement Track anchor, EEF20 min-max normalisation into the unified 80-D slots and the
templated UMT5 prompt.  The caller provides the pixels already in the training domain (see
``rt2_sim.observe``).

Sampling follows the training schedule (``r_T = r_A = 0.5, r_V = 1``) with 20 rounds: Track and
Action are final after round 10, where sampling stops unless the predicted video is wanted for a
debug video (the remaining rounds only refine Video; the action is identical either way).

Per-request ``policy`` options (default: none of them):
    rounds   total sampler rounds (Action is final after ``rounds / 2``)
    cfg      guidance scale on the Action expert's text + proprio shortcuts; the unconditional branch is the
             training-time shortcut dropout (both hidden): ``v = v_null + cfg (v_cond - v_null)``
    samples  K noise samples per request; the executed chunk is the medoid of the K normalised EEF20 chunks
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from metiswam4d.build import build_model, build_schedule
from metiswam4d.config import load_stage_config
from metiswam4d.data.rt2.codec import anchor_frame
from metiswam4d.data.rt2.eef import EEFNormalizer, gather_unified, scatter_unified
from metiswam4d.data.rt2.episode_dataset import multiview_layout
from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder, center_pad
from metiswam4d.data.rt2.text_cache import PromptTextCache
from metiswam4d.sampler import SampleConditions
from metiswam4d.train.checkpoint import load_model_weights

CAMERAS = ("head_camera", "left_camera", "right_camera")
MAX_FORWARD_ROWS = 8  # batch rows per model forward (the GPU is shared with 5 simulators)


def sample_seed(seed: int, k: int) -> int:
    """Noise seed of the k-th sample of a request; sample 0 keeps the request seed."""
    if k == 0:
        return int(seed) & 0x7FFFFFFFFFFFFFFF
    return int(hashlib.sha256(f"{int(seed)}:{k}".encode()).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


def take_conditions(cond: SampleConditions, idx: torch.Tensor) -> SampleConditions:
    t = lambda v: None if v is None else v[idx.to(v.device)]
    return dataclasses.replace(
        cond, context=t(cond.context), context_mask=t(cond.context_mask), video_first_frame=t(cond.video_first_frame),
        track_anchor=t(cond.track_anchor), condition_present=t(cond.condition_present), proprio=t(cond.proprio),
        proprio_mask=t(cond.proprio_mask), embodiment=t(cond.embodiment),
        track_conditions=None if cond.track_conditions is None else tuple(t(c) for c in cond.track_conditions))


def rot6d_to_quat_xyzw(r6: np.ndarray) -> np.ndarray:
    """Inverse of ``eef.quat_xyzw_to_rot6d`` (rot6d = first two rotation-matrix columns), Gram-Schmidt."""
    from scipy.spatial.transform import Rotation
    r6 = np.asarray(r6, dtype=np.float64)
    a, b = r6[..., :3], r6[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return Rotation.from_matrix(np.stack((x, y, z), axis=-1)).as_quat()


def eef20_to_sim(eef20: np.ndarray) -> np.ndarray:
    """``[N, 20]`` EEF20 -> ``[N, 16]`` RoboTwin ``take_action(action_type="ee")`` (xyz, quat, gripper per arm).

    The quaternion keeps the numeric order that ``read_eef20`` consumed from ``endpose``."""
    arms = []
    for start in (0, 10):
        arm = eef20[..., start:start + 10]
        arms.append(np.concatenate((arm[..., :3], rot6d_to_quat_xyzw(arm[..., 3:9]),
                                    np.clip(arm[..., 9:10], 0.0, 1.0)), axis=-1))
    return np.concatenate(arms, axis=-1).astype(np.float64)


class RT2Policy:
    def __init__(self, config: str | Path, checkpoint: str | Path, device: str = "cuda:0", rounds: int = 20,
                 log=print):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        stage = load_stage_config(config)
        self.stage, self.spec = stage, stage.data.spec
        started = time.time()
        model = build_model(stage.model)
        report = load_model_weights(model, checkpoint)
        if report["missing"] or report["unexpected"]:
            raise RuntimeError(f"checkpoint does not cover the model: {report}")
        self.model = model.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        log(f"[policy] {checkpoint}: {report['loaded']} tensors ({time.time() - started:.0f}s)")
        schedule, _ = build_schedule(stage)
        self.schedule, self.rounds = schedule, rounds
        self.encoder = RT2OnlineEncoder(stage.data.rt2_encoder, self.device)
        self.eef = EEFNormalizer(stage.data.rt2.alpha_stats)
        self.embodiment = int(stage.data.rt2.embodiment)
        self.text_cache = PromptTextCache(Path(stage.data.rt2.root) / "text_cache")
        self._umt5 = None
        self._text: dict[str, torch.Tensor] = {}

    def text(self, prompt: str) -> torch.Tensor:
        """``[L, 4096]`` bf16 UMT5 features of a templated prompt: offline training cache, else online."""
        if prompt not in self._text:
            if prompt in self.text_cache:
                value = self.text_cache.load(prompt)
            else:
                # The 11 GB encoder lives on the CPU and visits the GPU only for a cache miss: the GPU is shared
                # with the simulators' renderers and cuRobo.
                if self._umt5 is None:
                    from metiswam4d.data.human.text import UMT5Online
                    self._umt5 = UMT5Online(self.stage.data.rt2_encoder.vae_path, torch.device("cpu"),
                                            max_len=self.spec.max_text_len, dtype=self.dtype)
                self._umt5.encoder.to(self.device)
                self._umt5.device = self.device
                try:
                    hidden, mask = self._umt5([prompt])
                    value = hidden[0, : int(mask[0].sum())].cpu()
                finally:
                    self._umt5.encoder.to("cpu")
                    self._umt5.device = torch.device("cpu")
                    torch.cuda.empty_cache()
            self._text[prompt] = value
        return self._text[prompt]

    def predict(self, request: dict) -> dict:
        return self.predict_batch([request])[0]

    @torch.no_grad()
    def predict_batch(self, requests: list[dict]) -> list[dict]:
        """Each request: ``rgb`` {camera: uint8 HxWx3} (training domain), ``depth_mm`` float HxW (head camera),
        ``mask`` bool HxW (robot | task objects), ``eef20`` [20] raw, ``prompt`` (templated), ``seed``.
        ``full`` (same for the whole batch) runs all rounds and decodes the predicted video / Track."""
        started = time.time()
        dev, dt = self.device, self.dtype
        b = len(requests)
        full = bool(requests[0].get("full", False))
        layouts = np.stack([multiview_layout({c: Image.fromarray(r["rgb"][c]) for c in CAMERAS}, CAMERAS, 384, 320)
                            for r in requests])
        video_first = self.encoder.encode_pixels(torch.from_numpy(layouts)[:, None])                  # [B, 48, 1, 24, 20]
        masks = np.stack([np.asarray(r["mask"], dtype=bool) for r in requests])
        anchor_px = center_pad(torch.from_numpy(np.stack([anchor_frame(m) for m in masks]))[:, None], 256, 320)
        track_anchor = self.encoder.encode_pixels(anchor_px)                                           # [B, 48, 1, 16, 20]
        cond_px = self.encoder.condition_pixels(
            torch.from_numpy(np.stack([r["rgb"]["head_camera"] for r in requests])),
            torch.from_numpy(np.stack([np.asarray(r["depth_mm"], np.float32) for r in requests])),
            torch.from_numpy(masks))
        cond = self.encoder.encode_pixels(cond_px)                                                     # [3B, 48, 1, 16, 20]
        unified, dims = scatter_unified(self.eef.normalize(np.stack([np.asarray(r["eef20"], np.float32)
                                                                      for r in requests])))
        texts = [self.text(r["prompt"]) for r in requests]
        length = max(t.shape[0] for t in texts)
        context = torch.zeros(b, length, texts[0].shape[1], dtype=torch.bfloat16)
        context_mask = torch.zeros(b, length, dtype=torch.bool)
        for i, t in enumerate(texts):
            context[i, :t.shape[0]], context_mask[i, :t.shape[0]] = t, True
        f = self.spec.track_frames
        conditions = SampleConditions(
            context=context.to(dev), context_mask=context_mask.to(dev),
            video_first_frame=video_first, video_future_frames=f - 1,
            track_anchor=track_anchor, track_future_frames=f - 1,
            track_conditions=(cond[:b], cond[b:2 * b], cond[2 * b:]),
            camera_frames=self.spec.frame_slots,
            action_horizon=self.spec.action_horizon, action_dim=self.spec.action_dim,
            proprio=torch.from_numpy(unified[:, None]).to(dev, dt),
            proprio_mask=torch.from_numpy(np.broadcast_to(dims, (b, 1, dims.shape[0])).copy()).to(dev),
            embodiment=torch.full((b,), self.embodiment, device=dev),
        )
        opts = requests[0].get("policy") or {}
        k = int(opts.get("samples", 1))
        cfg = float(opts.get("cfg", 1.0))
        rounds = int(opts.get("rounds", self.rounds))
        rows = torch.arange(b).repeat_interleave(k)                                                   # request of each row
        generators = [torch.Generator().manual_seed(sample_seed(requests[i]["seed"], j))
                      for i in range(b) for j in range(k)]
        per_forward = max(1, MAX_FORWARD_ROWS // (2 if cfg != 1.0 else 1))
        parts = []
        with torch.autocast(device_type="cuda", dtype=dt):
            for s in range(0, len(rows), per_forward):
                sl = slice(s, s + per_forward)
                part, rounds_run = self._sample(take_conditions(conditions, rows[sl]), generators[sl],
                                                stop_on_action=not full, rounds=rounds, cfg=cfg)
                parts.append(part)
        x = {m: torch.cat([p[m] for p in parts]) for m in parts[0]}
        norm = gather_unified(x["action"].float().cpu().numpy()).reshape(b, k, -1)                  # [B, K, 32*20]
        chosen, spread = [], []
        for i in range(b):
            dist = np.linalg.norm(norm[i][:, None] - norm[i][None], axis=-1)                         # [K, K]
            chosen.append(i * k + int(dist.sum(1).argmin()))
            spread.append(float(dist.sum() / max(k * (k - 1), 1)))
        seconds = time.time() - started
        outs = []
        for i in range(b):
            eef20 = self.eef.denormalize(gather_unified(x["action"][chosen[i]].float().cpu().numpy()))  # [32, 20]
            actions = eef20_to_sim(eef20)
            if not np.isfinite(actions).all():
                raise ValueError("non-finite predicted action")
            outs.append({"actions": actions, "eef20": eef20, "rounds_run": rounds_run, "batch": b,
                         "inference_seconds": seconds, "sample_spread": spread[i]})
        if full:
            pick = torch.as_tensor(chosen)
            video = self.encoder.decode_latents(x["video"][pick]).cpu().numpy()                     # [B, 9, 384, 320, 3]
            track = self.encoder.decode_latents(x["track"][pick]).cpu().numpy()                     # [B, 9, 256, 320, 3]
            for i in range(b):
                outs[i].update(pred_video=video[i], pred_track=track[i])
            del video, track
            torch.cuda.empty_cache()
        return outs

    def _sample(self, cond: SampleConditions, generators: list[torch.Generator], stop_on_action: bool,
                rounds: int, cfg: float = 1.0, hide_track: bool = False, hide_video: bool = False):
        """``AsyncSampler.sample`` with one noise generator per batch element (a sample's noise, drawn in the
        sampler's order video / track / camera / action, does not depend on what else is in the batch).
        ``cfg != 1`` evaluates each row twice per round (conditional, Action shortcuts hidden) and guides only the
        Action velocity; Video / Track / camera follow the conditional branch.  ``hide_track`` / ``hide_video`` set the
        training-time modality dropout flags (the future Track / Video branch is invisible to the other experts)."""
        from metiswam4d.model import ModelInput
        present = cond.present
        trajectory = self.schedule.present(present).inference_trajectory(rounds)
        dev, dt = cond.context.device, self.dtype
        streams: dict[str, list[torch.Tensor]] = {}

        def noise(name: str, shape) -> torch.Tensor:
            streams[name] = [torch.randn(shape, generator=g, dtype=torch.float32) for g in generators]
            return torch.stack(streams[name]).to(device=dev, dtype=dt)

        x: dict[str, torch.Tensor] = {}
        c, _, h, w = cond.video_first_frame.shape[1:]
        x["video"] = torch.cat((cond.video_first_frame.to(dt), noise("video", (c, cond.video_future_frames, h, w))), 2)
        c, _, h, w = cond.track_anchor.shape[1:]
        x["track"] = torch.cat((cond.track_anchor.to(dt), noise("track", (c, cond.track_future_frames, h, w))), 2)
        if cond.camera_frames > 0:
            x["camera"] = noise("camera", (cond.camera_frames, cond.camera_dim))
        x["action"] = noise("action", (cond.action_horizon, cond.action_dim))
        b = len(generators)
        guided = cfg != 1.0
        rep = 2 if guided else 1
        dup = (lambda v: None if v is None else torch.cat((v, v))) if guided else (lambda v: v)
        base = dict(context=dup(cond.context), context_mask=dup(cond.context_mask),
                    track_conditions=None if cond.track_conditions is None else tuple(dup(c) for c in cond.track_conditions),
                    condition_present=dup(cond.condition_present), proprio=dup(cond.proprio),
                    proprio_mask=dup(cond.proprio_mask), embodiment=dup(cond.embodiment))
        if guided:
            hidden = torch.cat((torch.zeros(b, dtype=torch.bool), torch.ones(b, dtype=torch.bool))).to(dev)
            base.update(drop_action_text=hidden, drop_action_proprio=hidden)
        if hide_track:
            base["drop_track"] = torch.ones(rep * b, dtype=torch.bool, device=dev)
        if hide_video:
            base["drop_video"] = torch.ones(rep * b, dtype=torch.bool, device=dev)
        sig = lambda v: torch.full((rep * b,), float(v), device=dev)
        rounds_run = 0
        for i in range(rounds):
            now = {m: trajectory[m][i].item() for m in present}
            nxt = {m: trajectory[m][i + 1].item() for m in present}
            if all(now[m] <= 0 for m in present):
                break
            out = self.model(ModelInput(
                video=dup(x["video"]), video_sigma=sig(now["video"]),
                track=dup(x["track"]), track_sigma=sig(now["track"]),
                camera=dup(x.get("camera")),
                action=dup(x["action"]), action_sigma=sig(now["action"]), **base))
            rounds_run = i + 1
            for m in present:
                if now[m] <= 0:
                    continue
                v = out.velocity(m).to(x[m].dtype)
                if guided:
                    v = v[b:] + cfg * (v[:b] - v[b:]) if m == "action" else v[:b]
                step = nxt[m] - now[m]
                if m in ("video", "track"):
                    x[m][:, :, 1:] = x[m][:, :, 1:] + step * v[:, :, 1:]
                    if m == "track" and "camera" in x and out.camera_velocity is not None:
                        x["camera"] = x["camera"] + step * out.camera_velocity[:b].to(x["camera"].dtype)
                else:
                    x[m] = x[m] + step * v
            if stop_on_action and nxt["action"] <= 0:
                break
        return x, rounds_run


__all__ = ["CAMERAS", "RT2Policy", "eef20_to_sim", "rot6d_to_quat_xyzw"]
