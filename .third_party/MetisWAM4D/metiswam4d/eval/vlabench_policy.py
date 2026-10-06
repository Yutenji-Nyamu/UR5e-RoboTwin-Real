"""MetisWAM4D policy for VLABench, served over the OpenWAM deploy WebSocket protocol.

The official OpenWAM VLABench client (``benchmarks/vlabench/single_eval.py``) talks to this server unchanged except
for ``scripts/vlabench/metis_single_eval.py``, which adds the Track conditions the model was trained with to every
observation message (``metis``: the head view resized 480 -> 240x320, the head z-depth in mm and the robot mask on the
240x320 stretched grid, all from the evaluator's own ``obs``).

Inputs follow ``VLABenchEpisodeDataset`` / ``RT2OnlineEncoder``: L layout 384x320 (the client already LANCZOS-resized
the views to their slots, so ``lshape_layout`` pastes them unchanged), head RGB / depth / robot mask condition images,
the zero-displacement anchor on the robot pixels, raw EEF10 proprio (base-frame xyz, rot6d, open flag) -> Alpha VLABench
min-max -> unified slots 0-9, templated UMT5 prompt.  Sampling is ``RT2Policy._sample`` (20 rounds, Action final after
round 10).  Output: 32 raw EEF10 actions (denormalized), of which ``execute_steps`` are executed before re-planning.

Protocol (``openwam/deploy/server.py``): ``obs`` -> ``{"type": "action", "action": [10 floats], ...}`` one control step at a
time from a buffer refilled by inference when empty; ``reset`` clears the buffer and starts a new episode; ``ping`` ->
``pong``.  The noise seed of a re-plan is a hash of (policy seed, episode index on this server, re-plan index), so a job
(one track x task, one client) is reproducible.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from collections import deque
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from metiswam4d.build import build_model, build_schedule
from metiswam4d.config import load_stage_config
from metiswam4d.data.rt2.codec import anchor_frame
from metiswam4d.data.rt2.eef import load_pickled_stats
from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder, center_pad
from metiswam4d.data.rt2.text_cache import PromptTextCache, format_prompt
from metiswam4d.data.vlabench.episode_dataset import EEF10, UNIFY_DIM, lshape_layout
from metiswam4d.eval.rt2_policy import MAX_FORWARD_ROWS, RT2Policy, sample_seed, take_conditions
from metiswam4d.sampler import SampleConditions
from metiswam4d.train.checkpoint import load_model_weights

GRID_H, GRID_W = 240, 320


def replan_seed(policy_seed: int, episode: int, replan: int) -> int:
    key = f"metiswam4d-vlabench:{int(policy_seed)}:{int(episode)}:{int(replan)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


def decode_png(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)


class VLABenchPolicy(RT2Policy):
    def __init__(self, config: str | Path, checkpoint: str | Path, device: str = "cuda:0", rounds: int = 20,
                 log=print):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        stage = load_stage_config(config)
        data = stage.data.robodojo
        if data is None or data.source != "vlabench":
            raise ValueError(f"{config} is not a VLABench stage config")
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
        eef = load_pickled_stats(data.alpha_stats)["eef"]
        self.lo = np.asarray(eef["min"], np.float32)[:EEF10]
        self.range = np.maximum(np.asarray(eef["max"], np.float32)[:EEF10] - self.lo, 1e-6)
        self.embodiment = int(data.embodiment)
        self.text_cache = PromptTextCache(Path(data.root) / "text_cache")
        self._umt5 = None
        self._text: dict[str, torch.Tensor] = {}
        self.stage.data.rt2_encoder = stage.data.robodojo_encoder   # RT2Policy.text() reads the UMT5 path here

    def normalize(self, eef10: np.ndarray) -> np.ndarray:
        out = np.zeros((*eef10.shape[:-1], UNIFY_DIM), np.float32)
        out[..., :EEF10] = 2.0 * (np.asarray(eef10, np.float32) - self.lo) / self.range - 1.0
        return out

    def denormalize(self, unified: np.ndarray) -> np.ndarray:
        return ((np.asarray(unified, np.float32)[..., :EEF10] + 1.0) / 2.0 * self.range + self.lo).astype(np.float32)

    @torch.no_grad()
    def predict_batch(self, requests: list[dict]) -> list[dict]:
        """Each request: ``views`` [head 256x320, wrist 128x160, right 128x160] uint8, ``head_track`` uint8 240x320,
        ``depth_mm`` float 240x320, ``mask`` bool 240x320 (robot), ``eef10`` [10] raw, ``prompt`` (templated),
        ``seed``, optional ``policy`` {rounds, cfg, samples, hide_track, contrast} and ``reference`` (normalized [n, 10]):
        of ``samples`` plans the medoid is returned, or with a reference the one minimizing
        coherence (distance of its first n steps to the reference) + ``contrast`` x mean distance to the others."""
        started = time.time()
        dev, dt = self.device, self.dtype
        b = len(requests)
        layouts = np.stack([lshape_layout([Image.fromarray(v) for v in r["views"]], 384, 320) for r in requests])
        video_first = self.encoder.encode_pixels(torch.from_numpy(layouts)[:, None])                  # [B, 48, 1, 24, 20]
        masks = np.stack([np.asarray(r["mask"], dtype=bool) for r in requests])
        anchor_px = center_pad(torch.from_numpy(np.stack([anchor_frame(m) for m in masks]))[:, None], 256, 320)
        track_anchor = self.encoder.encode_pixels(anchor_px)                                           # [B, 48, 1, 16, 20]
        cond_px = self.encoder.condition_pixels(
            torch.from_numpy(np.stack([r["head_track"] for r in requests])),
            torch.from_numpy(np.stack([np.asarray(r["depth_mm"], np.float32) for r in requests])),
            torch.from_numpy(masks))
        cond = self.encoder.encode_pixels(cond_px)                                                     # [3B, 48, 1, 16, 20]
        unified = self.normalize(np.stack([np.asarray(r["eef10"], np.float32) for r in requests]))
        dims = np.zeros(UNIFY_DIM, bool)
        dims[:EEF10] = True
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
            proprio_mask=torch.from_numpy(np.broadcast_to(dims, (b, 1, UNIFY_DIM)).copy()).to(dev),
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
                                                stop_on_action=True, rounds=rounds, cfg=cfg,
                                                hide_track=bool(opts.get("hide_track", False)))
                parts.append(part)
        action = torch.cat([p["action"] for p in parts]).float().cpu().numpy()[..., :EEF10]
        action = action.reshape(b, k, *action.shape[1:])                                            # [B, k, 32, 10]
        contrast = float(opts.get("contrast", 0.5))
        outs = []
        for i in range(b):
            flat = action[i].reshape(k, -1)
            score = np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(1) / max(k - 1, 1)
            reference = requests[i].get("reference")
            if reference is not None and k > 1:
                # backward coherence: the new plan's first steps against the unexecuted tail of the previous plan
                n = len(reference)
                coherence = np.linalg.norm((action[i][:, :n] - reference[None]).reshape(k, -1), axis=-1)
                score = coherence + contrast * score
            chosen = int(score.argmin())
            eef10 = self.denormalize(action[i, chosen])                                               # [32, 10]
            if not np.isfinite(eef10).all():
                raise ValueError("non-finite predicted action")
            outs.append({"eef10": eef10, "normalized": action[i, chosen], "rounds_run": rounds_run,
                         "inference_seconds": time.time() - started})
        return outs


class VLABenchServer:
    """One simulator client per server (the action buffer is per server, as in the official deploy).  With
    ``coherence`` in the options, each re-plan passes the unexecuted tail of the previous plan as the reference for
    choosing among the ``samples`` plans."""

    def __init__(self, policy: VLABenchPolicy, execute_steps: int = 32, policy_options: dict | None = None,
                 policy_seed: int = 42, log=print):
        self.policy = policy
        self.execute_steps = int(execute_steps)
        self.options = dict(policy_options or {})
        self.policy_seed = int(policy_seed)
        self.log = log
        self.buffer: deque = deque()
        self.tail = None
        self.episode, self.replan, self.steps = -1, 0, 0

    def reset(self) -> None:
        self.buffer.clear()
        self.tail = None
        self.episode += 1
        self.replan, self.steps = 0, 0

    def request(self, msg: dict) -> dict:
        images, extra = msg["images"], msg["metis"]
        depth = np.frombuffer(base64.b64decode(extra["depth_mm_f16"]), np.float16).reshape(GRID_H, GRID_W)
        mask = np.frombuffer(base64.b64decode(extra["mask_u8"]), np.uint8).reshape(GRID_H, GRID_W) > 0
        return {
            "views": [decode_png(images[k]) for k in ("head_camera", "left_wrist_camera", "right_wrist_camera")],
            "head_track": decode_png(extra["head_track"]),
            "depth_mm": depth.astype(np.float32), "mask": mask,
            "eef10": np.asarray(msg["state"], np.float32),
            "prompt": format_prompt(str(msg["prompt"])),
            "seed": replan_seed(self.policy_seed, max(self.episode, 0), self.replan),
            "policy": self.options,
        }

    def act(self, msg: dict) -> dict:
        started = time.time()
        if not self.buffer:
            request = self.request(msg)
            if self.options.get("coherence") and self.tail is not None and len(self.tail):
                request["reference"] = self.tail
            out = self.policy.predict(request)
            self.buffer.extend(out["eef10"][:self.execute_steps])
            self.tail = out["normalized"][self.execute_steps:]
            if self.replan == 0 and self.episode % 10 == 0:
                self.log(json.dumps({"episode": self.episode, "inference_seconds": round(out["inference_seconds"], 2),
                                     "rounds": out["rounds_run"]}))
            self.replan += 1
        action = self.buffer.popleft()
        self.steps += 1
        return {"action": [float(v) for v in action], "step": self.steps,
                "latency_ms": round((time.time() - started) * 1000, 2)}


def serve(server: VLABenchServer, host: str, port: int) -> None:
    import asyncio
    import websockets

    async def handler(ws):
        async for message in ws:
            try:
                data = json.loads(message)
                kind = data.get("type", "obs")
                if kind == "reset":
                    server.reset()
                    reply = {"type": "reset_ack"}
                elif kind == "ping":
                    reply = {"type": "pong"}
                elif kind == "obs":
                    reply = {"type": "action", **server.act(data)}
                else:
                    reply = {"type": "error", "code": "unknown_message_type", "message": f"unknown type {kind}"}
            except Exception as exc:  # noqa: BLE001 - reported to the client, which fails the episode loudly
                import traceback
                traceback.print_exc()
                reply = {"type": "error", "code": "internal_error", "message": repr(exc)[:500]}
            await ws.send(json.dumps(reply))

    async def main():
        async with websockets.serve(handler, host, port, max_size=None, ping_interval=None):
            print(f"MODEL_READY ws://{host}:{port}", flush=True)
            await asyncio.Future()

    asyncio.run(main())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage3_vlabench_v1.yaml")
    ap.add_argument("--checkpoint", required=True, help="bf16 state_dict (scripts/eval/export_bf16.py) or DCP dir")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--execute-steps", type=int, default=32)
    ap.add_argument("--policy", default="{}", help="JSON sampler options (rounds, cfg, samples, hide_track, coherence, contrast)")
    ap.add_argument("--policy-seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_num_threads(4)
    log = lambda m: print(m, flush=True)
    policy = VLABenchPolicy(args.config, args.checkpoint, log=log)
    for (prompt,) in policy.text_cache._connection().execute("SELECT prompt FROM embeddings").fetchall():
        policy.text(prompt)
    log(f"[policy] {len(policy._text)} cached prompts preloaded; execute_steps={args.execute_steps} "
        f"policy={args.policy} seed={args.policy_seed}")
    serve(VLABenchServer(policy, args.execute_steps, json.loads(args.policy), args.policy_seed, log=log),
          args.host, args.port)


if __name__ == "__main__":
    main()


__all__ = ["VLABenchPolicy", "VLABenchServer", "replan_seed"]
