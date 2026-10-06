"""Checkpointing that works for a plain module and for FSDP2-sharded models.

Layout::

    <output_dir>/checkpoints/step_000123/
        model/            torch.distributed.checkpoint (DCP) shards of the model state
        optimizer/        DCP shards of the optimizer state
        trainer.json      step, rng seeds, config snapshot
        complete.json     written last; a directory without it is incomplete

``load_model_weights`` accepts a DCP ``model`` directory, a checkpoint step
directory or a plain ``.pt`` state dict (weights-only warm start).
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch import nn


def _dcp():
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
        set_model_state_dict, set_optimizer_state_dict,
    )
    return dcp, StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def save_checkpoint(directory: Path, model: nn.Module, optimizer: torch.optim.Optimizer, *,
                    step: int, extra: dict | None = None, keep_last: int = 3) -> Path:
    directory = Path(directory)
    target = directory / f"step_{step:07d}"
    if is_main():
        target.mkdir(parents=True, exist_ok=True)
        (target / "model").mkdir(exist_ok=True)
        (target / "optimizer").mkdir(exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    dcp, Options, get_model, get_optim, _, _ = _dcp()
    if dist.is_initialized():
        dcp.save(get_model(model), checkpoint_id=str(target / "model"))
        dcp.save(get_optim(model, optimizer), checkpoint_id=str(target / "optimizer"))
    else:
        torch.save(model.state_dict(), target / "model" / "state.pt")
        torch.save(optimizer.state_dict(), target / "optimizer" / "state.pt")
    if is_main():
        (target / "trainer.json").write_text(json.dumps({
            "step": step, "time": time.time(), "torch": torch.__version__, **(extra or {})}, indent=2))
        (target / "complete.json").write_text(json.dumps({"step": step}))
        _rotate(directory, keep_last)
    if dist.is_initialized():
        dist.barrier()
    return target


def _rotate(directory: Path, keep_last: int) -> None:
    complete = sorted(p for p in directory.glob("step_*") if (p / "complete.json").exists())
    for old in complete[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


def latest_checkpoint(directory: Path) -> Path | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    complete = sorted(p for p in directory.glob("step_*") if (p / "complete.json").exists())
    return complete[-1] if complete else None


def _not_in_checkpoint(state: dict, checkpoint_dir: Path) -> list[str]:
    from torch.distributed.checkpoint import FileSystemReader
    metadata = FileSystemReader(str(checkpoint_dir)).read_metadata().state_dict_metadata
    return sorted(k for k in state if k not in metadata)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None,
                    log=None) -> int:
    """Resume the full training state.

    Parameters that the running code has but the checkpoint does not (modules added after the run started,
    e.g. ``track.condition_missing``) keep their fresh initialisation and a zero optimizer state; they are
    reported through ``log``.  Everything present in the checkpoint is restored exactly.
    """
    path = Path(path)
    if not (path / "complete.json").exists():
        raise FileNotFoundError(f"{path} is not a complete checkpoint")
    dcp, Options, get_model, get_optim, set_model, set_optim = _dcp()
    fresh: list[str] = []
    if dist.is_initialized() or (path / "model" / ".metadata").exists():
        from torch.distributed.checkpoint import DefaultLoadPlanner
        state = get_model(model)
        fresh = _not_in_checkpoint(state, path / "model")
        dcp.load(state, checkpoint_id=str(path / "model"), planner=DefaultLoadPlanner(allow_partial_load=True))
        set_model(model, state)
        if optimizer is not None and (path / "optimizer").exists():
            opt_state = get_optim(model, optimizer)
            dcp.load(opt_state, checkpoint_id=str(path / "optimizer"), planner=DefaultLoadPlanner(allow_partial_load=True))
            set_optim(model, optimizer, optim_state_dict=opt_state)
    else:
        saved = torch.load(path / "model" / "state.pt", map_location="cpu", weights_only=True)
        fresh = sorted(k for k in model.state_dict() if k not in saved)
        model.load_state_dict(saved, strict=False)
        if optimizer is not None and (path / "optimizer" / "state.pt").exists():
            opt_saved = torch.load(path / "optimizer" / "state.pt", map_location="cpu", weights_only=False)
            if fresh:  # the saved param groups predate the new parameters: restore per-parameter state only
                _load_optimizer_state_partial(optimizer, opt_saved, model, saved)
            else:
                optimizer.load_state_dict(opt_saved)
    if fresh and log is not None:
        log(f"[resume] {len(fresh)} parameters absent from the checkpoint keep their fresh init: {fresh[:8]}")
    return int(json.loads((path / "trainer.json").read_text())["step"])


def _load_optimizer_state_partial(optimizer: torch.optim.Optimizer, saved: dict, model: nn.Module, saved_model: dict) -> None:
    """Single-process optimizer restore when the model gained parameters: map saved per-parameter states onto the
    current parameters by name order within each group (saved groups are prefixes of the current ones)."""
    name_of = {p: n for n, p in model.named_parameters()}
    saved_groups = saved["param_groups"]
    for group, saved_group in zip(optimizer.param_groups, saved_groups):
        old_params = [p for p in group["params"] if name_of[p] in saved_model]
        for p, saved_id in zip(old_params, saved_group["params"]):
            if saved_id in saved["state"]:
                optimizer.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in saved["state"][saved_id].items()}
        for key, value in saved_group.items():
            if key != "params":
                group[key] = value


def load_model_weights(model: nn.Module, path: str | Path, prefixes: tuple[str, ...] | None = None,
                       exclude: tuple[str, ...] | None = None) -> dict:
    """Weights-only warm start (before FSDP wrapping).

    Shape-tolerant: tensors whose name is absent from the checkpoint or whose shape differs (e.g. the
    reading interface after changing ``num_queries``) keep their fresh initialisation and are reported.
    ``prefixes`` restricts the load to parameter names starting with one of them (e.g. only the Track
    expert and the reader of a previous stage while the Video expert comes from elsewhere); ``exclude``
    keeps the named sub-modules at their fresh initialisation (a head whose semantics changed).
    """
    path = Path(path)
    if path.name == "latest":
        resolved = latest_checkpoint(path.parent)
        if resolved is None:
            raise FileNotFoundError(f"no complete checkpoint under {path.parent}")
        path = resolved
    own = model.state_dict()
    if prefixes:
        own = {k: v for k, v in own.items() if k.startswith(tuple(prefixes))}
    if exclude:
        own = {k: v for k, v in own.items() if not k.startswith(tuple(exclude))}
    if path.is_dir():
        if (path / "model").exists():
            path = path / "model"
        if (path / "state.pt").exists():
            state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        else:
            from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
            dcp, *_ = _dcp()
            metadata = FileSystemReader(str(path)).read_metadata().state_dict_metadata
            state = {k: torch.empty_like(v, device="cpu") for k, v in own.items()
                     if k in metadata and tuple(getattr(metadata[k], "size", ())) == tuple(v.shape)}
            dcp.load(state, checkpoint_id=str(path), planner=DefaultLoadPlanner(allow_partial_load=True),
                     no_dist=True)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k in own]  # only report inside the requested prefixes
    return {"loaded": len(state), "missing": len(missing), "unexpected": len(unexpected),
            "missing_keys": sorted(missing)[:12]}


__all__ = ["is_main", "latest_checkpoint", "load_checkpoint", "load_model_weights", "save_checkpoint"]
