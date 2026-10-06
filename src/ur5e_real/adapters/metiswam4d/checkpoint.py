"""Explicit full local export; source/config/statistics travel with the weights."""

import hashlib
from pathlib import Path

import torch

from .contract import FPS, HORIZON, VERSION
from .native import build_model
from .policy import ActionPolicy


def parameter_digest(model, *, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    return parameter_digest(model, trainable=False)


def save(path, model, config, contract, source_sha256, *, vae_path=None):
    trainable_dtype = {p.dtype for p in model.parameters() if p.requires_grad}
    if len(trainable_dtype) != 1:
        raise ValueError("trainable parameters must use one consistent dtype")
    with Path(path).open("xb") as handle:
        torch.save({"format": VERSION, "config": config, "contract": contract,
                    "source_sha256": source_sha256, "vae_path": str(vae_path) if vae_path else None,
                    "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
                    "trainable_dtype": str(trainable_dtype.pop()).removeprefix("torch."),
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, handle)


def load(path, source_sha256, *, device="cpu", allow_tiny=False):
    value = torch.load(path, map_location="cpu", weights_only=True)
    contract = value["contract"]
    if (value["format"] != VERSION or contract.get("version") != VERSION
            or contract.get("fps") != FPS or contract.get("horizon") != HORIZON
            or contract.get("slots") != list(range(10, 17))):
        raise ValueError("Metis joint/RGB contract mismatch")
    if value["source_sha256"] != source_sha256:
        raise ValueError("local Metis source differs from the checkpoint source")
    if value["config"]["tiny"] and not allow_tiny:
        raise ValueError("tiny-smoke weights are diagnostic and cannot be used as a trained robot policy")
    if value["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("unsupported checkpoint dtype")
    model = build_model(value["config"]).to(dtype=getattr(torch, value["dtype"]))
    trainable_dtype = value.get("trainable_dtype", value["dtype"])
    if trainable_dtype not in {"float32", "bfloat16"}:
        raise ValueError("unsupported trainable checkpoint dtype")
    model.action.to(dtype=getattr(torch, trainable_dtype))
    model.proprio_encoder.to(dtype=getattr(torch, trainable_dtype))
    model.load_state_dict(value["model"], strict=True)
    model.to(device).eval()
    return ActionPolicy(model, contract["stats"]), value
