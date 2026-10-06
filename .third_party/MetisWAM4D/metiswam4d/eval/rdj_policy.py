"""MetisWAM4D policy for RoboDojo (ARX X5): one training-domain observation -> 32 EE actions.

Inputs follow ``RDJEpisodeDataset`` / ``RT2OnlineEncoder`` exactly: 3-view L layout 384x320, robot-only head depth
(mm) and robot mask as the Track conditions, the zero-displacement anchor on the robot pixels, EEF20 world -> per-arm
base frame (``world_to_base_eef20``) -> Alpha RoboDojo min-max -> unified 80-D slots, templated UMT5 prompt (offline
RDJ cache, online encoder for evaluation-only instructions).  Sampling is ``RT2Policy._sample`` (coupled clock,
20 rounds, Action final after round 10).

The websocket-served ``RDJServer`` answers one ``act(obs)`` call per re-plan (observation in, list of ``take_action``
dictionaries out) so several simulator clients can share one server: the official ``PolicyServer`` serialises model
calls, and a single call carries no cross-request state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from metiswam4d.build import build_model, build_schedule
from metiswam4d.config import load_stage_config
from metiswam4d.data.robodojo.eef_base import base_to_world_eef20, world_to_base_eef20
from metiswam4d.data.rt2.codec import anchor_frame
from metiswam4d.data.rt2.eef import EEFNormalizer, gather_unified, scatter_unified
from metiswam4d.data.rt2.episode_dataset import multiview_layout
from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder, center_pad
from metiswam4d.data.rt2.text_cache import PromptTextCache
from metiswam4d.eval.rdj_observation import CAMERAS, RobotGeometry, eef20_to_action_dicts, request_from_observation
from metiswam4d.eval.rt2_policy import MAX_FORWARD_ROWS, RT2Policy, sample_seed, take_conditions
from metiswam4d.sampler import SampleConditions
from metiswam4d.train.checkpoint import load_model_weights


def episode_seed(policy_seed: int, variant: str, layout_id: int) -> int:
    # Policy labels identify report columns, not different random draws. Preserve
    # the bare task-config seed used by the original 420-episode reference.
    variant = variant.split("@", 1)[0]
    return int(hashlib.sha256(f"metiswam4d-rdj:{int(policy_seed)}:{variant}:{int(layout_id)}".encode()).hexdigest()[:16], 16)


def replan_seed(seed: int, replan: int) -> int:
    return int(hashlib.sha256(f"{int(seed)}:replan:{int(replan)}".encode()).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


class RDJPolicy(RT2Policy):
    def __init__(self, config: str | Path, checkpoint: str | Path, device: str = "cuda:0", rounds: int = 20,
                 log=print):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        stage = load_stage_config(config)
        if stage.data.robodojo is None or stage.data.robodojo_encoder is None:
            raise ValueError(f"{config} is not a RoboDojo stage config")
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
        self.encoder = RT2OnlineEncoder(stage.data.robodojo_encoder, self.device)
        self.eef = EEFNormalizer(stage.data.robodojo.alpha_stats)
        self.embodiment = int(stage.data.robodojo.embodiment)
        self.text_cache = PromptTextCache(Path(stage.data.robodojo.root) / "text_cache")
        self._umt5 = None
        self._text: dict[str, torch.Tensor] = {}
        # RT2Policy.text() reads the VAE path from the RT2 encoder config; alias it.
        self.stage.data.rt2_encoder = stage.data.robodojo_encoder

    @torch.no_grad()
    def predict_batch(self, requests: list[dict]) -> list[dict]:
        """Each request: ``rgb`` {camera: uint8 240x320x3}, ``depth_mm`` float 240x320 (robot pixels, 0 elsewhere),
        ``mask`` bool 240x320 (robot), ``eef20`` [20] world frame, ``prompt`` (templated), ``seed``, optional
        ``policy`` {rounds, cfg, samples} and ``full`` (decode the predicted video / Track)."""
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
        eef_world = np.stack([np.asarray(r["eef20"], np.float32) for r in requests])
        unified, dims = scatter_unified(self.eef.normalize(world_to_base_eef20(eef_world)))
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
        rows = torch.arange(b).repeat_interleave(k)
        generators = [torch.Generator().manual_seed(sample_seed(requests[i]["seed"], j))
                      for i in range(b) for j in range(k)]
        per_forward = max(1, MAX_FORWARD_ROWS // (2 if cfg != 1.0 else 1))
        parts = []
        with torch.autocast(device_type="cuda", dtype=dt):
            for s in range(0, len(rows), per_forward):
                sl = slice(s, s + per_forward)
                part, rounds_run = self._sample(take_conditions(conditions, rows[sl]), generators[sl],
                                                stop_on_action=not full, rounds=rounds, cfg=cfg,
                                                hide_track=bool(opts.get("hide_track", False)),
                                                hide_video=bool(opts.get("hide_video", False)))
                parts.append(part)
        x = {m: torch.cat([p[m] for p in parts]) for m in parts[0]}
        norm = gather_unified(x["action"].float().cpu().numpy()).reshape(b, k, -1)                  # [B, K, 32*20]
        chosen, spread = [], []
        for i in range(b):
            dist = np.linalg.norm(norm[i][:, None] - norm[i][None], axis=-1)
            chosen.append(i * k + int(dist.sum(1).argmin()))
            spread.append(float(dist.sum() / max(k * (k - 1), 1)))
        seconds = time.time() - started
        outs = []
        for i in range(b):
            base = self.eef.denormalize(gather_unified(x["action"][chosen[i]].float().cpu().numpy()))  # [32, 20] base frame
            eef20 = base_to_world_eef20(base)
            if not np.isfinite(eef20).all():
                raise ValueError("non-finite predicted action")
            outs.append({"eef20": eef20, "eef20_base": base.astype(np.float32), "rounds_run": rounds_run, "batch": b,
                         "inference_seconds": seconds, "sample_spread": spread[i]})
        if full:
            pick = torch.as_tensor(chosen)
            video = self.encoder.decode_latents(x["video"][pick]).cpu().numpy()
            track = self.encoder.decode_latents(x["track"][pick]).cpu().numpy()
            for i in range(b):
                outs[i].update(pred_video=video[i], pred_track=track[i])
            del video, track
            torch.cuda.empty_cache()
        return outs


def blend_chunks(prev: np.ndarray, executed: int, cur: np.ndarray, weight: float) -> np.ndarray:
    """Temporal ensembling of two consecutive EEF20 chunks: step ``j`` of the new chunk coincides with step
    ``executed + j`` of the previous one; the overlap is averaged with weight ``weight`` on the previous prediction
    (decaying linearly to 0 at the end of the overlap, so the new chunk takes over smoothly).  Gripper included."""
    cur = np.asarray(cur, np.float64).copy()
    tail = np.asarray(prev, np.float64)[executed:]
    n = min(len(tail), len(cur))
    if n <= 0 or weight <= 0:
        return cur.astype(np.float32)
    w = weight * (1.0 - np.arange(n) / n)[:, None]
    cur[:n] = w * tail[:n] + (1.0 - w) * cur[:n]
    return cur.astype(np.float32)


class RDJServer:
    """The object served by XPolicyLab's ``PolicyServer``: ``act(obs)`` -> ``execute_steps`` action dictionaries."""

    def __init__(self, config: str, checkpoint: str, output: Path, execute_steps: int = 32, policy: dict | None = None,
                 policy_seed: int = 42, save_first_observation: bool = True, log=print):
        self.policy = RDJPolicy(config, checkpoint, log=log)
        # The 35 training prompts are read from the cache once here (the server thread pool would otherwise touch
        # the sqlite connection from several threads); evaluation-only instructions go to the online UMT5.
        cache = self.policy.text_cache
        for (prompt,) in cache._connection().execute("SELECT prompt FROM embeddings").fetchall():
            self.policy.text(prompt)
        log(f"[policy] {len(self.policy._text)} cached prompts preloaded")
        self.geometry = RobotGeometry()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.execute_steps = int(execute_steps)
        self.options = dict(policy or {})
        self.policy_seed = int(policy_seed)
        self.save_first_observation = save_first_observation
        self.log = log
        self.calls = 0

    def reset(self):
        return {"ok": True}

    def act(self, obs: dict) -> dict:
        """``obs``: official observation + ``task``, ``variant``, ``layout_id``, ``replan`` (0-based re-plan index),
        optional ``policy`` (per-request options overriding the campaign's), ``prev_eef20`` / ``executed`` (the previous
        chunk and how many of its steps were executed, for temporal ensembling)."""
        variant, layout, replan = str(obs.get("variant", obs.get("task"))), int(obs.get("layout_id", 0)), int(obs.get("replan", 0))
        options = {**self.options, **(obs.get("policy") or {})}
        seed = replan_seed(episode_seed(self.policy_seed, variant, layout), replan)
        request = request_from_observation(obs, self.geometry, seed, options)
        if self.save_first_observation and replan == 0:
            path = self.output / "observations" / f"{variant}_layout{layout:03d}.npz"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, head=request["rgb"]["head_camera"], left=request["rgb"]["left_camera"],
                                    right=request["rgb"]["right_camera"], depth_mm=request["depth_mm"],
                                    mask=request["mask"], eef20=request["eef20"], prompt=request["prompt"],
                                    raw_head=np.asarray(obs["vision"]["cam_head"]["color"]))
        out = self.policy.predict(request)
        eef20 = out["eef20"]
        ensemble = float(options.get("ensemble", 0.0))
        if ensemble > 0 and obs.get("prev_eef20") is not None and replan > 0:
            eef20 = blend_chunks(np.asarray(obs["prev_eef20"], np.float32), int(obs["executed"]), eef20, ensemble)
        execute = int(options.get("execute_steps", self.execute_steps))
        actions = eef20_to_action_dicts(eef20)[:execute]
        self.calls += 1
        if self.calls % 20 == 1:
            self.log(json.dumps({"variant": variant, "layout": layout, "replan": replan,
                                 "inference_seconds": round(out["inference_seconds"], 2), "rounds": out["rounds_run"]}))
        return {"actions": actions, "eef20": out["eef20"].astype(np.float32),
                "inference_seconds": float(out["inference_seconds"]), "rounds_run": int(out["rounds_run"])}


def main() -> None:
    import argparse
    import asyncio
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="campaign.json")
    ap.add_argument("--output", required=True, help="per-GPU output directory (first observations, log)")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    campaign = json.loads(Path(args.campaign).read_text())
    root = Path(campaign["robodojo_root"])
    sys.path.extend([str(root), str(root / "XPolicyLab")])
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    torch.set_num_threads(4)
    server = RDJServer(campaign["config"], campaign["model_file"], Path(args.output),
                       execute_steps=campaign["protocol"]["execute_steps"], policy=campaign["protocol"].get("policy"),
                       policy_seed=campaign["protocol"]["policy_seed"],
                       log=lambda m: print(m, flush=True))
    print("MODEL_READY", flush=True)
    asyncio.run(PolicyServer(server, PolicyServerConfig(host="127.0.0.1", port=args.port)).serve_forever())


if __name__ == "__main__":
    main()


__all__ = ["RDJPolicy", "RDJServer", "episode_seed", "replan_seed"]
