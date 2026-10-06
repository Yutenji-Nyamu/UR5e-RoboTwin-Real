"""Load the predecessor's trained Track branch (JanusTrack Fusion-v3, DCP) into ``TrackExpert``.

The v3 ``V2TrackBranch`` is a Wan-style expert with different attribute names; the computation is
identical (same AdaLN order, RMS-normed Q/K, 3-axis RoPE split 22/21/21 pairs with the same
frequencies, cosine-then-sine time embedding, ``[condition | anchor | future]`` token layout).

    model.track_branch.expert.blocks.N.context_attn.*   -> blocks.N.cross_attn.*
    model.track_branch.expert.blocks.N.norm2.*          -> blocks.N.norm3.*      (affine norm before cross-attn)
    model.track_branch.expert.time_embedding.*          -> time_embedding.*
    model.track_branch.expert.time_projection.*         -> time_projection.*
    model.track_branch.expert.final_modulation          -> head.modulation
    model.track_branch.head.*                           -> head.head.*
    everything else (patch_embedding, condition_*, role_embedding, text_embedding, self_attn, ffn,
    modulation) keeps its name.
"""
from __future__ import annotations

from pathlib import Path
import re

import torch

from metiswam4d.experts.local import TrackExpert

PREFIX = "model.track_branch."


def v3_key_to_local(key: str) -> str:
    key = key.removeprefix(PREFIX)
    key = re.sub(r"^expert\.blocks\.(\d+)\.context_attn\.", r"blocks.\1.cross_attn.", key)
    key = re.sub(r"^expert\.blocks\.(\d+)\.norm2\.", r"blocks.\1.norm3.", key)
    key = re.sub(r"^expert\.blocks\.", "blocks.", key)
    key = re.sub(r"^expert\.time_embedding\.", "time_embedding.", key)
    key = re.sub(r"^expert\.time_projection\.", "time_projection.", key)
    if key == "expert.final_modulation":
        return "head.modulation"
    if key in ("head.weight", "head.bias"):
        return "head.head." + key.split(".")[1]
    return key


def local_key_to_v3(key: str) -> str:
    if key == "head.modulation":
        return PREFIX + "expert.final_modulation"
    if key.startswith("head.head."):
        return PREFIX + "head." + key.split(".")[-1]
    key = re.sub(r"^blocks\.(\d+)\.cross_attn\.", r"blocks.\1.context_attn.", key)
    key = re.sub(r"^blocks\.(\d+)\.norm3\.", r"blocks.\1.norm2.", key)
    if key.startswith(("blocks.", "time_embedding.", "time_projection.")):
        key = "expert." + key
    return PREFIX + key


@torch.no_grad()
def load_v3_track(track: TrackExpert, checkpoint_dir: str | Path, *, strict: bool = True) -> dict:
    """``checkpoint_dir`` is the DCP step directory (containing ``pytorch_model_fsdp_0``) or that subfolder."""
    from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader, load

    folder = Path(checkpoint_dir)
    if (folder / "pytorch_model_fsdp_0").exists():
        folder = folder / "pytorch_model_fsdp_0"
    metadata = FileSystemReader(str(folder)).read_metadata().state_dict_metadata
    source_keys = [k for k in metadata if k.startswith(PREFIX)]
    state = track.state_dict()
    request: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for src in source_keys:
        dst = v3_key_to_local(src)
        if dst not in state:
            unexpected.append(src)
            continue
        if tuple(metadata[src].size) != tuple(state[dst].shape):
            raise RuntimeError(f"shape mismatch {src} {tuple(metadata[src].size)} vs {dst} {tuple(state[dst].shape)}")
        request[src] = torch.empty(tuple(metadata[src].size), dtype=state[dst].dtype)
    # Modules that post-date the v3 checkpoint keep their fresh init and are not counted as missing.
    fresh = ("camera_embedding.", "camera_type", "camera_head.", "condition_missing")
    missing = sorted(k for k in state if local_key_to_v3(k) not in request and not k.startswith(fresh))
    if strict and (missing or unexpected):
        raise RuntimeError(f"v3 Track load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    load(request, checkpoint_id=str(folder), planner=DefaultLoadPlanner(allow_partial_load=True), no_dist=True)
    for src, tensor in request.items():
        state[v3_key_to_local(src)].copy_(tensor.to(state[v3_key_to_local(src)].dtype))
    return {"loaded": len(request), "missing": missing, "unexpected": unexpected}


__all__ = ["load_v3_track", "local_key_to_v3", "v3_key_to_local"]
