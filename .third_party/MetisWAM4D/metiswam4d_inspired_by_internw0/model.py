"""Model assembly for the Memory / Open variant.

* Video expert: Wan2.2-TI2V-5B DiT from InternW0-Delta (``mixtures.video.*``, identical names; the Causal-Imprint
  ``future_delta_*`` tensors are not used).  Its clean prefix is ``MEMORY_SLOTS + 1`` latent frames: the five memory
  frames and the current observation, temporal positions consecutive as in InternW0's online inference.
* Action expert: InternW0's ActionDiT (``mixtures.action.*``; ``head`` -> ``action_decoder``).  InternW0 replaced every
  action cross-attention by one reading raw RynnBrain hidden states (K / V of width 2048); without the VLM the
  cross-attention K / V / norm_k and the text embedding come from OpenWAM-Alpha-Sim-RoboDojo's UMT5 path (same
  1024 / 3072 shapes), Q / O stay InternW0's.
* proprio encoder: InternW0's video-context proprio projection (its ``type_embedding`` folded into the bias).
* Track expert, reader, progress heads, embodiment table: RoboDojo v6 (same Track codec and head-camera grid).
* Action reads the clean world only (``action_read = none``): the memory + current video tokens and the Track
  conditions, as InternW0's action reads its prefilled clean frames.  Future Video / Track are denoised as
  co-trained world-model targets that shape the shared representations.
"""
from __future__ import annotations

from pathlib import Path
import re

import torch
from torch import Tensor

from metiswam4d.build import build_model
from metiswam4d.config import ModelConfig
from metiswam4d.experts.alpha import action_key_from_alpha, find_checkpoint
from metiswam4d.model import MetisWAM4D, ModelInput
from metiswam4d.schedule import AsyncSchedule

from metiswam4d_inspired_by_internw0.data import MEMORY_SLOTS

CLEAN_PREFIX = MEMORY_SLOTS + 1
INTERNW0_CKPT = "/ytech_milm_intern/danglingwei/model_zoos/InternW0-Delta/InternW0-Delta-RoboDojo/robodojo.pt"
ALPHA_RDJ = "/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/OpenWAM-Alpha-Sim-RoboDojo"
V6_WEIGHTS = ("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/stage3_robodojo_v6_focus_r2/simeval_focus/"
              "checkpoint/model_bf16_tail3.pt")
V6_PREFIXES = ("track.", "reader.", "progress_", "embodiment_embedding.")
ACTION_TEXT_PATH = re.compile(r"^(blocks\.\d+\.cross_attn\.(k|v|norm_k)\.|text_embedding\.)")


def _strip_memory(module, args, kwargs):
    """Forward pre-hook on the reader: it sees the current window only (one clean frame + futures)."""
    hidden, grid = kwargs.get("video_hidden"), kwargs.get("video_grid")
    if hidden is not None and grid is not None and grid[0] > CLEAN_PREFIX - 1:
        per_frame = grid[1] * grid[2]
        kwargs["video_hidden"] = hidden[:, MEMORY_SLOTS * per_frame:]
        kwargs["video_grid"] = (grid[0] - MEMORY_SLOTS, grid[1], grid[2])
    return args, kwargs


def build_iw0_model(cfg: ModelConfig) -> MetisWAM4D:
    model = build_model(cfg)
    if model.video is not None:
        model.video.config.clean_frames = CLEAN_PREFIX
    if model.reader is not None:
        model.reader.register_forward_pre_hook(_strip_memory, with_kwargs=True)
    return model


@torch.no_grad()
def _copy(target: dict[str, Tensor], name: str, value: Tensor, report: dict) -> None:
    if name not in target:
        report["unexpected"].append(name)
        return
    if tuple(value.shape) != tuple(target[name].shape):
        report["mismatched"][name] = (tuple(value.shape), tuple(target[name].shape))
        return
    target[name].copy_(value.to(dtype=target[name].dtype, device=target[name].device))
    report["loaded"].add(name)


@torch.no_grad()
def load_internw0(model: MetisWAM4D, checkpoint: str = INTERNW0_CKPT, alpha: str = ALPHA_RDJ) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    mot = payload["mot"]
    out: dict = {}
    if model.video is not None:
        target, rep = model.video.state_dict(), {"loaded": set(), "unexpected": [], "mismatched": {}}
        for key, value in mot.items():
            if key.startswith("mixtures.video.") and ".future_delta" not in key:
                _copy(target, key[len("mixtures.video."):], value, rep)
        rep["missing"] = sorted(set(target) - rep["loaded"])
        out["video"] = rep
    if model.action is not None:
        target, rep = model.action.state_dict(), {"loaded": set(), "unexpected": [], "mismatched": {}}
        for key, value in mot.items():
            if not key.startswith("mixtures.action."):
                continue
            name = key[len("mixtures.action."):]
            if ACTION_TEXT_PATH.match(name):
                continue
            _copy(target, re.sub(r"^head\.", "action_decoder.", name), value, rep)
        from safetensors import safe_open
        with safe_open(str(find_checkpoint(alpha)), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("action_backbone."):
                    name = action_key_from_alpha(key[len("action_backbone."):])
                    if ACTION_TEXT_PATH.match(name):
                        _copy(target, name, handle.get_tensor(key), rep)
        rep["missing"] = sorted(set(target) - rep["loaded"])
        out["action"] = rep
        proprio = payload["proprio_encoder"]
        model.proprio_encoder.weight.copy_(proprio["weight"].to(model.proprio_encoder.weight.dtype))
        model.proprio_encoder.bias.copy_((proprio["bias"].float() + proprio["type_embedding"].float())
                                         .to(model.proprio_encoder.bias.dtype))
        out["proprio"] = {"loaded": 2}
    return {k: ({**v, "loaded": len(v["loaded"])} if isinstance(v.get("loaded"), set) else v) for k, v in out.items()}


def initialize_iw0(model: MetisWAM4D, *, internw0: str = INTERNW0_CKPT, alpha: str = ALPHA_RDJ,
                   v6: str | None = V6_WEIGHTS, log=print) -> dict:
    report = {"internw0": load_internw0(model, internw0, alpha)}
    for name in ("video", "action"):
        r = report["internw0"].get(name)
        if r:
            log(f"[init] InternW0 {name}: {r['loaded']} tensors, missing {len(r['missing'])} {r['missing'][:6]}, "
                f"mismatched {len(r['mismatched'])}, unexpected {len(r['unexpected'])}")
            if r["missing"] or r["mismatched"]:
                raise RuntimeError(f"InternW0 {name} load incomplete: {r}")
    if v6:
        from metiswam4d.train.checkpoint import load_model_weights
        report["v6"] = load_model_weights(model, v6, prefixes=V6_PREFIXES)
        log(f"[init] v6 Track / reader / progress / embodiment from {v6}: {report['v6']}")
    return report


class ActionSampler:
    """Action-only flow sampling against the clean world (``action_read = none``): the Video expert sees the memory
    frames + the current frame, the Track expert its conditions + anchor, both at sigma 0; the action block follows
    its own shifted trajectory for ``steps`` Euler steps."""

    def __init__(self, schedule: AsyncSchedule, steps: int = 10):
        self.shift = float(schedule.shift["action"])
        self.steps = int(steps)

    def sigmas(self, device) -> Tensor:
        from metiswam4d.schedule import flow_shift
        return flow_shift(torch.linspace(1.0, 0.0, self.steps + 1, device=device, dtype=torch.float64), self.shift)

    @torch.no_grad()
    def sample(self, model: MetisWAM4D, *, context: Tensor, context_mask: Tensor | None, video_clean: Tensor,
               track_anchor: Tensor | None, track_conditions: tuple[Tensor, ...] | None, proprio: Tensor,
               proprio_mask: Tensor | None, embodiment: Tensor, action_horizon: int, action_dim: int,
               generator: torch.Generator | None = None) -> Tensor:
        b, device, dtype = context.shape[0], context.device, video_clean.dtype
        x = torch.randn((b, action_horizon, action_dim), generator=generator, dtype=torch.float32).to(device, dtype)
        sig = self.sigmas(device)
        zero = torch.zeros(b, device=device)
        read_layers, model.read_layers = model.read_layers, ()   # no futures to read; Action never sees the reader
        try:
            for i in range(self.steps):
                s_now = float(sig[i])
                inputs = ModelInput(
                    context=context, context_mask=context_mask, video=video_clean, video_sigma=zero,
                    track=track_anchor, track_sigma=zero if track_anchor is not None else None,
                    track_conditions=track_conditions if track_anchor is not None else None,
                    action=x, action_sigma=torch.full((b,), s_now, device=device),
                    proprio=proprio, proprio_mask=proprio_mask, embodiment=embodiment)
                v = model(inputs).action_velocity.to(x.dtype)
                x = x + (float(sig[i + 1]) - s_now) * v
        finally:
            model.read_layers = read_layers
        return x


__all__ = ["ActionSampler", "CLEAN_PREFIX", "build_iw0_model", "initialize_iw0", "load_internw0"]
