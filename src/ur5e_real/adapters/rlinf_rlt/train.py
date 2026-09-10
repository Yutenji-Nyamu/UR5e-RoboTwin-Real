"""Bounded offline learners. No stage is accepted automatically from a loss value."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .actions import ActionCodec
from .cache import load_cache
from . import checkpoint
from .learner import RLTTrainer, td_target
from .run import load_run, update_state
from .storage import append_json, atomic_json, read_json
from .vendor.rlt_token_transformer import RLTTokenEncoder, RLTTokenTransformer


def resources():
    import psutil

    memory = psutil.virtual_memory()
    result = {"rss_bytes": psutil.Process().memory_info().rss, "ram_available_bytes": memory.available}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(
            cuda_free_bytes=free,
            cuda_total_bytes=total,
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
        )
    return result


def sample_cache(directory, cache, indices, device):
    prefix, masks = [], []
    for i in indices:
        with np.load(directory / "cache" / cache["files"][int(i)]["file"], allow_pickle=False) as sample:
            prefix.append(sample["prefix"])
            masks.append(sample["mask"])
    return torch.as_tensor(np.stack(prefix), device=device), torch.as_tensor(np.stack(masks), device=device)


@torch.no_grad()
def token_diagnostics(model, prefix, mask):
    model.eval()
    z = model.encode(prefix, mask)
    target = prefix.float()
    weights = mask[..., None].float()
    denominator = (weights.sum() * target.shape[-1]).clamp_min(1)

    def error(tokens):
        return float(((model.decode(tokens, prefix, mask).float() - target).square() * weights).sum() / denominator)

    actual, zero, shuffled = error(z), error(torch.zeros_like(z)), error(z.roll(1, 0))
    return {
        "reconstruction_mse": actual,
        "zero_token_mse": zero,
        "shuffle_token_mse": shuffled,
        "zero_gap": zero - actual,
        "shuffle_gap": shuffled - actual,
        "z_std": float(z.float().std()),
        "diagnostic_samples": len(prefix),
        "diagnostic_scope": "fixed training observations; not held-out generalization",
    }


def train_token(directory, *, steps=None, device="cuda", resume=True):
    directory, run, state = load_run(directory)
    if state["stage"] != "token" or state["selected_token"]:
        raise ValueError("token training is allowed only before freezing the representation")
    cache = load_cache(directory)
    cfg = run["config"]["token_train"]
    steps = cfg["steps"] if steps is None else steps
    if steps < 1 or len(cache["files"]) < 2:
        raise ValueError("need positive update budget and at least two observations for shuffle diagnostics")
    torch.manual_seed(cfg["seed"])
    generator = np.random.default_rng(cfg["seed"])
    model = RLTTokenTransformer(**run["config"]["token"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg["weight_decay"]
    )
    previous = 0
    if resume and state["latest_token"]:
        saved = checkpoint.load(state["latest_token"], run, cache, kind="token", device=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        previous = saved["metadata"]["step"]
        checkpoint.restore_rng(saved["rng"], generator)
    # Fixed evenly-spaced observations include different parts of the collected trajectories.
    diagnostic_ids = np.unique(np.linspace(0, len(cache["files"]) - 1, min(4, len(cache["files"])), dtype=int))
    diag_prefix, diag_mask = sample_cache(directory, cache, diagnostic_ids, device)
    path = None

    def diagnose(step):
        metrics = {
            "run_id": run["run_id"],
            "step": step,
            **token_diagnostics(model, diag_prefix, diag_mask),
            **resources(),
        }
        append_json(directory / "token_diagnostics.jsonl", metrics)
        atomic_json(directory / "token_diagnostics.json", metrics)
        print(f"[TOKEN DIAG] {metrics}", flush=True)

    diagnose(previous)
    for step in range(previous + 1, previous + steps + 1):
        started = time.monotonic()
        model.train()
        prefix, mask = sample_cache(
            directory, cache, generator.integers(len(cache["files"]), size=cfg["batch_size"]), device
        )
        # Explicit local short-fit schedule: warmup then cosine to 10% at the original 500-step budget.
        warmup = cfg["warmup_steps"]
        progress = min(1, max(0, (step - warmup) / max(1, cfg["steps"] - warmup)))
        multiplier = step / warmup if warmup and step <= warmup else 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(prefix, mask)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        RLTTrainer._finite(loss, norm)
        optimizer.step()
        append_json(
            directory / "token_train.jsonl",
            {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(norm),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - started,
                **resources(),
            },
        )
        if step % cfg["diagnostic_interval"] == 0 or step == previous + steps:
            diagnose(step)
        if step % cfg["save_interval"] == 0 or step == previous + steps:
            path = checkpoint.save(
                directory,
                "token",
                step,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": checkpoint.rng_state(generator),
                },
                cache_id=cache["cache_id"],
            )
    return path


def load_encoder(directory, run, cache, reference, device):
    saved = checkpoint.load(reference, run, cache, kind="token", device="cpu")
    encoder = RLTTokenEncoder(**run["config"]["token"])
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in saved["model"].items() if k.startswith("encoder.")}
    )
    return encoder.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encoded_cache(directory, run, cache, token, device):
    path = directory / "cache" / f"encoded_{token['sha256']}.npz"
    if not path.exists():
        encoder = load_encoder(directory, run, cache, token, device)
        codec = ActionCodec.from_dict(cache["codec"])
        data = {k: [] for k in ("z", "proprio", "reference", "observation_q")}
        for i, entry in enumerate(cache["files"]):
            prefix, mask = sample_cache(directory, cache, [i], device)
            data["z"].append(encoder(prefix, mask).flatten(1)[0].cpu().numpy())
            with np.load(directory / "cache" / entry["file"], allow_pickle=False) as sample:
                data["proprio"].append(sample["proprio"])
                data["reference"].append(codec.encode(sample["raw_reference"])[: run["config"]["real"]["action_steps"]])
                data["observation_q"].append(sample["observation_q"])
        from .storage import atomic_npz

        atomic_npz(path, **{k: np.stack(v) for k, v in data.items()})
        from .storage import file_digest

        atomic_json(
            str(path) + ".json",
            {"cache_id": cache["cache_id"], "token_sha256": token["sha256"], "sha256": file_digest(path)},
        )
    from .storage import file_digest, read_json

    if read_json(str(path) + ".json") != {
        "cache_id": cache["cache_id"],
        "token_sha256": token["sha256"],
        "sha256": file_digest(path),
    }:
        raise ValueError("encoded cache identity mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


@torch.no_grad()
def bc_diagnostics(trainer, data, codec):
    trainer.actor.eval()
    prediction = []
    for start in range(0, len(data["z"]), 32):
        batch = {k: torch.as_tensor(v[start : start + 32], device=trainer.device) for k, v in data.items()}
        prediction.append(
            trainer.actor(batch["z"], batch["proprio"], batch["reference"], deterministic=True).cpu().numpy()
        )
    prediction = np.concatenate(prediction)
    actual = np.stack([codec.decode(a, q) for a, q in zip(prediction, data["observation_q"], strict=True)])
    expected = np.stack([codec.decode(a, q) for a, q in zip(data["reference"], data["observation_q"], strict=True)])
    error = np.abs(actual - expected)
    return {
        "canonical_mse": float(np.mean((prediction - data["reference"]) ** 2)),
        "joint_mae_rad": float(error[..., :6].mean()),
        "joint_p95_rad": float(np.quantile(error[..., :6], 0.95)),
        "joint_max_rad": float(error[..., :6].max()),
        "gripper_mae": float(error[..., 6].mean()),
        "gripper_binary_agreement": float(((actual[..., 6] >= 0.5) == (expected[..., 6] >= 0.5)).mean()),
        "gripper_close_threshold_agreement": float(((actual[..., 6] >= 0.6) == (expected[..., 6] >= 0.6)).mean()),
        "gripper_open_threshold_agreement": float(((actual[..., 6] <= 0.4) == (expected[..., 6] <= 0.4)).mean()),
        "diagnostic_scope": "all offline training observations, deterministic deployed actor; not physical success",
    }


@torch.no_grad()
def replay_diagnostics(trainer, replay):
    """Full replay TD/Q and known terminal-return calibration, not just the last minibatch."""
    predictions, errors = [], []
    cfg = trainer.config["algorithm"]
    trainer.actor.eval()
    for start in range(0, len(replay["z"]), 32):
        batch = {k: torch.as_tensor(v[start : start + 32], device=trainer.device) for k, v in replay.items()}
        q = trainer.critic(batch["z"], batch["proprio"], batch["action"])
        action = trainer.actor(batch["next_z"], batch["next_proprio"], batch["next_reference"], deterministic=True)
        target_q = trainer.target(batch["next_z"], batch["next_proprio"], action).min(-1).values
        target = td_target(
            batch["rewards"],
            batch["lengths"],
            batch["terminated"],
            batch["truncated"],
            target_q,
            cfg["gamma"],
            cfg["bootstrap_on_timeout"],
        )
        predictions.append(q.cpu().numpy())
        errors.append((q - target[:, None]).cpu().numpy())
    q, error = np.concatenate(predictions), np.concatenate(errors)
    returns = np.full(len(q), np.nan)
    following = np.nan
    for i in range(len(q) - 1, -1, -1):
        if replay["terminated"][i]:
            following = 0.0
        elif replay["truncated"][i]:
            following = np.nan  # Missing the continuation; do not pretend the return is zero.
        length = int(replay["lengths"][i])
        reward = sum(float(r) * cfg["gamma"] ** j for j, r in enumerate(replay["rewards"][i, :length]))
        following = reward + cfg["gamma"] ** length * following
        returns[i] = following
    known = np.isfinite(returns)
    return {
        "replay_td_mae": float(np.abs(error).mean()),
        "replay_td_mse": float(np.square(error).mean()),
        "q_quantiles": np.quantile(q, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "terminal_return_mae": float(np.abs(q[known] - returns[known, None]).mean()) if known.any() else None,
        "known_return_samples": int(known.sum()),
        "positive_return_samples": int((returns > 0).sum()),
        "timeout_return_samples_excluded": int((~known).sum()),
    }


def train_heads(directory, *, mode, steps, device="cuda"):
    directory, run, state = load_run(directory)
    if steps < 1 or mode not in ("bc", "warmup", "online"):
        raise ValueError("supply a positive bounded update budget and bc/warmup/online mode")
    allowed = {"bc": ("bc",), "warmup": ("reference", "actor_probe"), "online": ("online",)}
    if state["stage"] not in allowed[mode]:
        raise ValueError("learner mode differs from the accepted stage")
    if mode != "bc":
        from .replay import load_replay

        summary = read_json(directory / "rounds" / (state["round"] or "missing") / "summary.json")
        if summary["attempts"] != 4 or summary["pending"]:
            raise ValueError("complete and label all four attempts before updating A/C")
        replay, replay_identity = load_replay(directory)
        if not state["review_pending"]:
            raise ValueError("finish a fixed-policy round before A/C updates")
    cache = load_cache(directory)
    token = state["selected_token"]
    if not token:
        raise ValueError("select a frozen token checkpoint first")
    torch.manual_seed(run["config"]["token_train"]["seed"])
    generator = np.random.default_rng(run["config"]["token_train"]["seed"])
    trainer = RLTTrainer(run["config"], device)
    # Always resume the latest *paired* learner state; execution selects a version separately.
    if state["latest_head"]:
        saved = checkpoint.load(
            state["latest_head"], run, cache, kind="heads", token_sha256=token["sha256"], device=device
        )
        trainer.load_state_dict(saved["learner"])
        checkpoint.restore_rng(saved["rng"], generator)
    offline = encoded_cache(directory, run, cache, token, device)
    codec = ActionCodec.from_dict(cache["codec"])
    path = None
    for step in range(steps):
        started = time.monotonic()
        source = offline if mode == "bc" else replay
        indices = generator.integers(len(source["z"]), size=run["config"]["algorithm"]["batch_size"])
        batch = {k: v[indices] for k, v in source.items() if k != "observation_q"}
        if mode == "bc":
            # Executor history is unobserved in SFT demos. Teach the reference mimic to ignore
            # nuisance auxiliary inputs; never invent these values for Q/reward replay.
            batch["proprio"] = batch["proprio"].copy()
            phase = np.eye(3, dtype=np.float32)[generator.integers(3, size=len(indices))]
            batch["proprio"][:, 7:] = np.concatenate([phase, generator.random((len(indices), 3))], axis=-1)
        trainer.actor.train()
        metrics = trainer.bc_step(batch) if mode == "bc" else trainer.update(batch, phase=mode)
        total = trainer.bc_updates + trainer.critic_updates
        payload = {
            "learner": trainer.state_dict(),
            "rng": checkpoint.rng_state(generator),
            "replay_identity": None if mode == "bc" else replay_identity,
        }
        path = checkpoint.save(
            directory, "heads", total, payload, cache_id=cache["cache_id"], token_sha256=token["sha256"]
        )
        checkpoint.prune_heads(directory)
        append_json(
            directory / "heads_train.jsonl",
            {
                "mode": mode,
                "step": total,
                **metrics,
                "checkpoint": str(path),
                "elapsed_s": time.monotonic() - started,
                **resources(),
            },
        )
        if (step + 1) % 100 == 0 or step + 1 == steps:
            diagnostic = {
                "run_id": run["run_id"],
                "step": total,
                "mode": mode,
                "checkpoint": str(path),
                "cache_id": cache["cache_id"],
                **bc_diagnostics(trainer, offline, codec),
                "last_update": metrics,
                "replay_identity": None if mode == "bc" else replay_identity,
            }
            diagnostic["executor_aux_probes"] = []
            for auxiliary in ([1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0]):
                probe = dict(offline)
                probe["proprio"] = offline["proprio"].copy()
                probe["proprio"][:, 7:] = auxiliary
                diagnostic["executor_aux_probes"].append({"aux": auxiliary, **bc_diagnostics(trainer, probe, codec)})
            diagnostic["noise_scale_max_slope_times_std"] = (
                codec.scale * (codec.q99 - codec.q01 + 1e-6) / 2 * run["config"]["algorithm"]["fixed_std"]
            ).tolist()
            if mode != "bc":
                diagnostic.update(replay_diagnostics(trainer, replay))
                diagnostic["gradient_updates_per_replay_transition_this_call"] = steps / len(replay["z"])
            append_json(directory / "heads_diagnostics.jsonl", diagnostic)
            atomic_json(directory / "heads_diagnostics.json", diagnostic)
            print(
                f"[HEADS DIAG] mode={mode} update={total} joint_mae={diagnostic['joint_mae_rad']:.5g} "
                f"gripper_agreement={diagnostic['gripper_binary_agreement']:.4f}; details={directory / 'heads_diagnostics.json'}",
                flush=True,
            )
    update_state(directory, review_pending=True)
    return path
