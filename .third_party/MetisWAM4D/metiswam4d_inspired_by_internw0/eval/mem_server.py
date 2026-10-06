"""Policy server for the Memory / Open model (``/usr/bin/python3.10``), same RPCs as ``iw0_server``
(``begin`` / ``act`` / ``close``, one session per simulator episode, acknowledgements per executed action).

Per re-plan at step ``t`` the model sees, exactly as in training:
    video     memory canvases of steps [0, t-128, t-96, t-64, t-32] (clamped to >= 0) + the current canvas, each
              canvas built from the live RGB through the corpus JPEG path (``training_rgb``) and InternW0's canvas
    track     robot-only SAPIEN depth / mask of the live joint state and head camera, RGB condition, anchor frame
    action    ``ActionSampler`` (clean world, 10 Euler steps) -> 32 joint targets, the first ``execute_steps``
              executed; the client ships the observations of re-plan steps only (= the memory steps for 32)

    /usr/bin/python3.10 -m metiswam4d_inspired_by_internw0.eval.mem_server --port P --output DIR --overrides '{...}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
ROBODOJO = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo")


class MemServer:
    def __init__(self, config: str, weights: str, output: Path, execute_steps: int = 32, seed: int = 42,
                 text_cache: str | None = None, log=print):
        import yaml
        from metiswam4d.build import build_schedule
        from metiswam4d.config import load_stage_config
        from metiswam4d.data.rt2.text_cache import PromptTextCache
        from metiswam4d.eval.rdj_observation import RobotGeometry
        from metiswam4d_inspired_by_internw0.data import IW0OnlineEncoder
        from metiswam4d_inspired_by_internw0.joint_action import JointNormalizer
        from metiswam4d_inspired_by_internw0.model import ActionSampler, build_iw0_model
        self.device = torch.device("cuda")
        self.stage = load_stage_config(config)
        self.model = build_iw0_model(self.stage.model)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"weights miss {len(missing)} tensors: {missing[:8]}")
        self.model = self.model.to(self.device, torch.bfloat16).eval()
        self.encoder = IW0OnlineEncoder(self.stage.data.robodojo_encoder, self.device)
        schedule, _ = build_schedule(self.stage)
        self.sampler = ActionSampler(schedule, steps=int(self.stage.schedule.inference_rounds))
        root = Path(self.stage.data.robodojo.root)
        self.text_cache = PromptTextCache(Path(text_cache) if text_cache else root / "text_cache")
        self._umt5 = None
        self._text: dict[str, torch.Tensor] = {}
        self.geometry = RobotGeometry()
        self.joints = JointNormalizer()
        self.execute_steps, self.seed = int(execute_steps), int(seed)
        self.sessions: dict[str, dict] = {}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log, self.calls = log, 0
        log(f"[mem_server] {weights}: {len(state)} tensors, unexpected {len(unexpected)}; execute {self.execute_steps}")

    # -- conditions ----------------------------------------------------------------------------------------------
    def text(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text:
            if prompt in self.text_cache:
                self._text[prompt] = self.text_cache.load(prompt)
            else:
                if self._umt5 is None:
                    from metiswam4d.data.human.text import UMT5Online
                    self._umt5 = UMT5Online(self.stage.data.robodojo_encoder.vae_path, self.device)
                hidden, mask = self._umt5([prompt])
                self._text[prompt] = hidden[0, : int(mask[0].sum())].to("cpu", torch.bfloat16)
        return self._text[prompt]

    def canvas(self, obs: dict) -> np.ndarray:
        from metiswam4d.eval.rdj_observation import LIVE_CAMERAS, training_rgb
        from metiswam4d_inspired_by_internw0.data import internw0_canvas
        return internw0_canvas(*[training_rgb(obs["vision"][c]["color"]) for c in LIVE_CAMERAS])

    # -- RPCs ----------------------------------------------------------------------------------------------------
    def reset(self):
        return {"ok": True}

    def begin(self, payload: dict):
        self.sessions[str(payload["session"])] = {"step": 0, "canvases": {}, "replan": 0}
        return self.info()

    def info(self):
        return {"replan_steps": self.execute_steps, "action_horizon": 32, "recent_capture_mod": 0}

    def close(self, payload: dict):
        self.sessions.pop(str(payload["session"]), None)
        return {"ok": True}

    @torch.no_grad()
    def act(self, payload: dict) -> dict:
        from metiswam4d.data.rt2.codec import anchor_frame
        from metiswam4d.data.rt2.online_encoder import center_pad
        from metiswam4d.data.rt2.text_cache import format_prompt
        from metiswam4d.eval.rdj_observation import instruction_of, joint_vector, training_rgb
        from metiswam4d_inspired_by_internw0.data import memory_indices
        from metiswam4d_inspired_by_internw0.joint_action import joints_to_action_dicts, scatter, unified_to_joints
        name = str(payload["session"])
        s = self.sessions.setdefault(name, {"step": 0, "canvases": {}, "replan": 0})
        acks = list(payload["acks"])
        if s["canvases"]:
            s["step"] += len(acks)
        for offset, obs in enumerate(acks):
            if obs is not None:
                step = s["step"] - (len(acks) - 1 - offset)
                s["canvases"][step] = self.canvas(obs)
        obs, t = acks[-1], s["step"]
        mem = [s["canvases"].get(i, s["canvases"][max(k for k in s["canvases"] if k <= i)]) for i in memory_indices(t)]
        frames = np.stack(mem + [s["canvases"][t]])                                        # [6, 384, 256, 3]
        t0 = time.time()
        video = self.encoder.encode_pixels(torch.from_numpy(frames)[:, None]).squeeze(2)   # [6, 48, 24, 16]
        video = video.permute(1, 0, 2, 3)[None]                                             # [1, 48, 6, 24, 16]
        depth_mm, mask = self.geometry(obs)
        head = torch.from_numpy(training_rgb(obs["vision"]["cam_head"]["color"]))[None]
        cond = self.encoder.encode_pixels(self.encoder.condition_pixels(
            head, torch.from_numpy(depth_mm.astype(np.float32))[None], torch.from_numpy(mask)[None]))
        anchor = center_pad(torch.from_numpy(anchor_frame(mask))[None, None], self.encoder.config.track_height,
                            self.encoder.config.track_width)
        track_anchor = self.encoder.encode_pixels(anchor)
        prompt = format_prompt(instruction_of(obs))
        context = self.text(prompt).to(self.device)[None]
        proprio, dims = scatter(self.joints.normalize(joint_vector(obs), "state")[None])
        generator = torch.Generator().manual_seed(zlib.crc32(f"{self.seed}:{name.split(':', 1)[-1]}:{s['replan']}".encode()))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            u80 = self.sampler.sample(
                self.model, context=context.to(torch.bfloat16), context_mask=None, video_clean=video,
                track_anchor=track_anchor, track_conditions=(cond[0:1], cond[1:2], cond[2:3]),
                proprio=torch.from_numpy(proprio[None]).to(self.device, torch.bfloat16),
                proprio_mask=torch.from_numpy(dims[None, None].copy()).to(self.device),
                embodiment=torch.tensor([self.stage.data.robodojo.embodiment], device=self.device),
                action_horizon=32, action_dim=80, generator=generator)
        joints = unified_to_joints(u80[0].float().cpu().numpy(), self.joints)
        seconds = time.time() - t0
        s["replan"] += 1
        self.calls += 1
        if self.calls % 50 == 1:
            self.log(json.dumps({"session": name, "step": t, "seconds": round(seconds, 2), "prompt": prompt[-80:]}))
        return {"actions": joints_to_action_dicts(joints[: self.execute_steps]), "joint": joints, "step": t,
                "inference_seconds": seconds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overrides", default="{}", help='JSON {"config", "weights", "execute_steps", "seed"}')
    args = ap.parse_args()
    o = json.loads(args.overrides)
    sys.path.extend([str(ROBODOJO), str(ROBODOJO / "XPolicyLab")])
    torch.set_num_threads(4)
    server = MemServer(o["config"], o["weights"], Path(args.output), execute_steps=int(o.get("execute_steps", 32)),
                       seed=int(o.get("seed", 42)), text_cache=o.get("text_cache"), log=lambda m: print(m, flush=True))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    print("MODEL_READY", json.dumps(server.info()), flush=True)
    asyncio.run(PolicyServer(server, PolicyServerConfig(host="127.0.0.1", port=args.port)).serve_forever())


if __name__ == "__main__":
    main()
