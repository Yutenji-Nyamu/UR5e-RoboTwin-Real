"""OpenWAM-Alpha (official release) as a policy of the RT2 closed-loop campaign.

The model is served in-process by the official deploy stack (``OpenWAM_Official_260913``): ``configs/deploy.yaml``
unchanged (10 sync denoising steps, DiT velocity cache, torch.compile, prompt-embedding cache) ->
``build_server_from_config`` -> ``PolicyServer``.  A request is what the official RoboTwin client
(``benchmarks/robotwin/openwam2robotwin_interface.py``) sends: the three raw rendered RGB frames, the RoboTwin deploy
prompt template around the instruction and the 20-D EEF state from ``endpose``.  Every request resets the server and
drains one generated chunk through the official executor (``inference_horizon: null`` = the whole chunk), i.e. the
actions the official client would receive step by step until its next replan.  The noise seed is the server
default (42) for every chunk, as in the official deployment.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
import time

import numpy as np

OPENWAM = Path("/m2v_intern_v3/danglingwei/codes/OpenWAM_Official_260913")
ALPHA_RT2 = Path("/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/"
                 "OpenWAM-Alpha-Sim-RoboTwin-Full")
CLIENT_CAMERAS = {"head_camera": "head_camera", "left_camera": "left_wrist_camera",
                  "right_camera": "right_wrist_camera"}


def official_path() -> None:
    for path in (OPENWAM / "third_party", OPENWAM):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))


def alpha_observe(obs: dict) -> dict:
    """The official client's payload fields from a live observation (no training-domain transforms)."""
    official_path()
    from benchmarks.utils import action_conversion
    cams = obs["observation"]
    endpose = obs["endpose"]
    return {
        "rgb": {c: np.ascontiguousarray(cams[c]["rgb"], dtype=np.uint8) for c in CLIENT_CAMERAS},
        "state": action_conversion.robotwin_endpose_to_eef20d(endpose["left_endpose"], endpose["right_endpose"],
                                                              endpose["left_gripper"], endpose["right_gripper"]),
    }


class PromptEmbedLRU(OrderedDict):
    """``prompt -> text embedding`` LRU with the official size.  The official ``_BoundedPromptEmbedCache`` overrides
    ``__getitem__``, which ``OrderedDict.popitem`` calls on eviction: the 33rd distinct prompt raises ``KeyError``."""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.maxsize:
            super().__delitem__(next(iter(self)))


class AlphaRT2Policy:
    def __init__(self, ckpt_dir: str | Path = ALPHA_RT2, device: str = "cuda:0", log=print):
        official_path()
        import torch
        from omegaconf import OmegaConf
        from openwam.deploy.server import _load_deploy_yaml, _validate_inference_config, build_server_from_config
        from benchmarks.robotwin.prompt_template import format_prompt_for_inference
        from benchmarks.utils import action_conversion
        self._prompt, self._conversion = format_prompt_for_inference, action_conversion
        started = time.time()
        cfg = _load_deploy_yaml()
        _validate_inference_config(cfg)
        self.server = build_server_from_config(cfg=cfg, ckpt_dir=str(ckpt_dir), device=device)
        self.server._init_policy()
        engine = self.server.engine
        engine._prompt_embed_cache = PromptEmbedLRU(int(self.server.cfg.optimization.prompt_embed_cache.maxsize))
        inf = self.server.cfg.inference
        self.settings = {"denoise_steps": int(inf.denoise_steps), "denoise_mode": str(inf.denoise_mode),
                         "inference_horizon": OmegaConf.select(self.server.cfg, "inference.inference_horizon"),
                         "num_frames": int(inf.num_frames), "dit_cache": OmegaConf.to_container(
                             self.server.cfg.optimization.dit_cache),
                         "compile": bool(self.server.cfg.optimization.compile.enabled)}
        log(f"[alpha] {ckpt_dir} ready in {time.time() - started:.0f}s; {self.settings}; "
            f"cuda mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        return [self.predict(r) for r in requests]

    def predict(self, request: dict) -> dict:
        from PIL import Image
        started = time.time()
        obs = {"images": {CLIENT_CAMERAS[c]: Image.fromarray(img) for c, img in request["rgb"].items()},
               "prompt": self._prompt(request["instruction"]),
               "state": [float(v) for v in np.asarray(request["state"], np.float32).reshape(-1)]}
        self.server.reset()
        obs = self.server._obs_preprocessor.preprocess(obs)
        policy = self.server._policy
        eef20 = [policy.predict_action(obs)]
        while policy._executor._action_buffer:
            eef20.append(policy.predict_action(obs))
        eef20 = np.stack([np.asarray(a, dtype=np.float32) for a in eef20])
        actions = np.stack([self._conversion.eef20d_to_ee16d(a) for a in eef20]).astype(np.float64)
        if not np.isfinite(actions).all():
            raise ValueError("non-finite predicted action")
        out = {"actions": actions, "eef20": eef20, "rounds_run": self.settings["denoise_steps"], "batch": 1,
               "inference_seconds": time.time() - started}
        if request.get("full"):
            out.update(pred_video=np.zeros((9, 384, 320, 3), np.uint8), pred_track=np.zeros((9, 256, 320, 3), np.uint8))
        return out


__all__ = ["ALPHA_RT2", "AlphaRT2Policy", "alpha_observe", "official_path"]
