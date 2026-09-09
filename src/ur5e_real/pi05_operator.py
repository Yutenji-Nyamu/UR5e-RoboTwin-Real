"""DP-style operator commands; the native model remains in its isolated environment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace
import uuid

from .adapters.robotwin_pi05.client import PolicyClient
from .adapters.robotwin_pi05.config import NAME
from .adapters.robotwin_pi05.contract import read_contract, require_same_contract
from .adapters.robotwin_pi05.dataset import validate_dataset, write_json
from .adapters.robotwin_pi05.infer import DEFAULT_ACTION_STEPS, run as run_inference
from .adapters.robotwin_pi05.runtime import prepare
from .config import load_config
from .doctor import print_checks, run_doctor
from .operator import LAB_CONFIG, REPOSITORY, _enter_repository

# Explicit first-demo selection; init also accepts another checkpoint reference.
DEFAULT_CHECKPOINT = "joint5_sft_20260909_01:1000"
MODEL_PYTHON = REPOSITORY / ".venv" / "pi05" / "bin" / "python"


def resolve_checkpoint(value: str, root: Path | None = None) -> Path:
    root = root or REPOSITORY / "checkpoints" / "pi05" / NAME
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.parent != Path("."):
        raise FileNotFoundError(candidate)
    hint, separator, step = value.rpartition(":")
    if not separator:
        hint, step = "", value
    if not step.isascii() or not step.isdigit():
        raise ValueError("checkpoint must be a directory or [run_hint:]step, for example 20260909_01:1000")
    matches = [path.resolve() for path in sorted(root.glob(f"*/{step}")) if path.is_dir() and hint in path.parent.name]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint {value!r} matched {len(matches)} directories; pass an unambiguous run:step or path"
        )
    return matches[0]


def load_trial(value: str):
    lab = load_config(LAB_CONFIG)
    checkpoint = resolve_checkpoint(value)
    contract = read_contract(checkpoint / "ur5e_contract.json")
    dataset_id = contract["dataset_id"]
    parts = dataset_id.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("checkpoint dataset ID must be an org/name pair")
    data_root = (lab.collection.data_root / "pi05" / "lerobot").resolve()
    dataset = (data_root / dataset_id).resolve()
    if not dataset.is_relative_to(data_root):
        raise ValueError("checkpoint dataset path is outside the configured data root")
    actual, _ = validate_dataset(dataset)
    require_same_contract(contract, actual)
    verification = json.loads((checkpoint / "ur5e_verification.json").read_text())
    if not (checkpoint / "params").is_dir() or not all(
        verification.get(key) is True
        for key in ("frozen_unchanged", "trainable_changed", "checkpoint_reload", "optimizer_reload")
    ):
        raise ValueError("checkpoint lacks the final native audit; intermediate checkpoints are evaluate-only")
    return SimpleNamespace(
        lab=lab, checkpoint=checkpoint, dataset=dataset, contract=contract, verification=verification
    )


def infer_init() -> int:
    parser = argparse.ArgumentParser(
        prog="ur5e-pi05-infer-init", description="check devices, joint home, then open gripper"
    )
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true", help="resolve and print only; no hardware or model process")
    args = parser.parse_args()
    try:
        _enter_repository()
        trial = load_trial(args.checkpoint)
        print(f"[PI05 INIT] checkpoint={trial.checkpoint}; joint home from {trial.contract['dataset_id']}")
        if not args.dry_run and not print_checks(run_doctor(trial.lab, hardware=True)):
            return 1
        prepare(trial.lab, trial.contract, execute=not args.dry_run)
        return 0
    except KeyboardInterrupt:
        print("\n[STOPPED] joint initialization interrupted")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[BLOCKED] {exc}")
        return 2


def free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def stop_owned_process(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_ready(process, trial, port: int, instance_id: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model process exited with status {process.returncode}; inspect model.log")
        try:
            with PolicyClient(trial.contract, port=port, timeout_s=0.5) as client:
                if client.metadata.get("instance_id") != instance_id or client.metadata.get("checkpoint_path") != str(
                    trial.checkpoint
                ):
                    raise RuntimeError("port belongs to a different model instance; refusing to use it")
                return
        except (OSError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError("model startup/warmup timed out; inspect model.log")


@contextmanager
def model_server(trial, args, log_dir: Path):
    if not MODEL_PYTHON.is_file() or not os.access(MODEL_PYTHON, os.X_OK):
        raise FileNotFoundError(f"native model environment is missing: {MODEL_PYTHON}")
    port, instance_id = free_loopback_port(), uuid.uuid4().hex
    command = [
        str(MODEL_PYTHON),
        "-u",
        "-m",
        "ur5e_real.adapters.robotwin_pi05",
        "serve",
        "--dataset",
        str(trial.dataset),
        "--checkpoint",
        str(trial.checkpoint),
        "--port",
        str(port),
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--instance-id",
        instance_id,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPOSITORY / "src"), environment.get("PYTHONPATH"))))
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"[MODEL] loading and warming up; log={log_dir / 'model.log'}", flush=True)
    with (log_dir / "model.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_ready(process, trial, port, instance_id, args.startup_timeout)
            print("[MODEL READY] selected checkpoint verified; starting robot-side inference", flush=True)
            yield port, instance_id
        finally:
            stop_owned_process(process)


def infer() -> int:
    parser = argparse.ArgumentParser(prog="ur5e-pi05-infer", description="run pi05 with an automatically managed model")
    parser.add_argument("checkpoint", help="checkpoint directory or run:step, for example 20260909_01:1000")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--shadow", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="resolve and print only; no hardware/model process")
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS)
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument(
        "--speed", type=float, default=0.6, help="joint speed limit in rad/s, not a playback multiplier"
    )
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=float, default=1.5, help="live RPC deadline in seconds")
    parser.add_argument("--startup-timeout", type=float, default=600, help="model loading/warmup deadline in seconds")
    args = parser.parse_args()
    if not 1 <= args.action_steps <= 50 or args.chunks < 1 or not 1 <= args.diffusion_steps <= 50:
        parser.error("require K=1..50, chunks>=1, and diffusion steps=1..50")
    if not 0 < args.speed <= 0.6 or not 0 < args.timeout <= 1.5:
        parser.error("require joint speed in (0, 0.6] rad/s and RPC timeout in (0, 1.5] seconds")
    if not math.isfinite(args.startup_timeout) or not 0 < args.startup_timeout <= 1800:
        parser.error("startup timeout must be in (0, 1800] seconds")
    log_dir, result = None, {"status": "not_started"}
    try:
        _enter_repository()
        trial = load_trial(args.checkpoint)
        if args.execute and trial.verification.get("status") != "SFT_not_physical_validation":
            raise ValueError("development smoke checkpoints are not eligible for physical execution")
        print(f"[PI05] checkpoint={trial.checkpoint}; H=50 K={args.action_steps} 10Hz; speed<={args.speed}rad/s")
        if args.dry_run:
            print("[DRY RUN] no model process, cameras, gripper or robot connection")
            return 0
        new_log_dir = REPOSITORY / "logs" / "pi05_infer" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        new_log_dir.mkdir(parents=True, exist_ok=False)
        log_dir = new_log_dir
        write_json(
            log_dir / "run.json",
            {**vars(args), "checkpoint_path": str(trial.checkpoint), "dataset": str(trial.dataset)},
        )
        with model_server(trial, args, log_dir) as (port, instance_id):
            run_inference(
                SimpleNamespace(
                    dataset=trial.dataset,
                    lab_config=LAB_CONFIG,
                    port=port,
                    timeout=args.timeout,
                    mode="execute" if args.execute else "shadow",
                    action_steps=args.action_steps,
                    chunks=args.chunks,
                    speed=args.speed,
                    shadow_gripper=0.0,
                    run_id=None,
                    index=0,
                    output=log_dir / "execution.json",
                    expected_instance_id=instance_id,
                )
            )
        result = {"status": "completed", "automatic_success_judgement": False}
        return 0
    except KeyboardInterrupt:
        result = {"status": "interrupted"}
        print("\n[STOPPED] inference interrupted; owned model process cleaned up")
        return 130
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[BLOCKED] {exc}")
        return 2
    finally:
        if log_dir is not None:
            write_json(log_dir / "result.json", result)
            print(f"[LOGS] {log_dir}")
