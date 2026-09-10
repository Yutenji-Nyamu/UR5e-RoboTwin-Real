"""One staged command; native JAX, PyTorch learning and hardware remain isolated."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

from .operator import REPOSITORY, LAB_CONFIG, _enter_repository
from .adapters.rlinf_rlt.run import initialize, load_run, resolve_run, decide, begin_round
from .adapters.rlinf_rlt.storage import lease, read_json, append_json, file_digest, digest

RLT_PYTHON = REPOSITORY / ".venv" / "rlt" / "bin" / "python"
NATIVE_PYTHON = REPOSITORY / ".venv" / "pi05" / "bin" / "python"


def implementation_identity():
    root = REPOSITORY / "src" / "ur5e_real"
    files = [
        *sorted((root / "adapters" / "rlinf_rlt").rglob("*.py")),
        root / "adapters" / "robotwin_pi05" / "rlt_features.py",
        root / "rlt_operator.py",
        REPOSITORY / "integrations" / "rlt" / "requirements.lock",
    ]
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "source_sha256": digest({str(p.relative_to(REPOSITORY)): file_digest(p) for p in files}),
        "python": sys.version,
        "interpreter": sys.executable,
    }


@contextmanager
def launch_service(python, module, arguments, trial, log_dir, name, *, timeout=600):
    from .pi05_operator import free_loopback_port, stop_owned_process, wait_ready

    if not python.is_file():
        raise FileNotFoundError(f"missing isolated interpreter: {python}")
    port, instance = free_loopback_port(), uuid.uuid4().hex
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    # Leave room for the separate frozen token encoder; never allocate JAX's default 75% pool.
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = [str(python), "-u", "-m", module, *map(str, arguments), "--port", str(port), "--instance-id", instance]
    with (log_dir / f"{name}_{instance}.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=REPOSITORY, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            print(f"[RLT] warming {name}; log={log.name}", flush=True)
            wait_ready(process, trial, port, instance, timeout)
            yield port, instance
        finally:
            stop_owned_process(process)


@contextmanager
def round_policy(directory, run, round_dir):
    from .adapters.robotwin_pi05.client import PolicyClient

    trial = SimpleNamespace(contract=run["contract"], checkpoint=Path(run["checkpoint"]))
    with ExitStack() as stack:
        feature_port, feature_instance = stack.enter_context(
            launch_service(
                NATIVE_PYTHON,
                "ur5e_real.adapters.robotwin_pi05.rlt_features",
                ["--dataset", run["dataset"], "--checkpoint", run["checkpoint"]],
                trial,
                round_dir,
                "features",
            )
        )
        port, _ = stack.enter_context(
            launch_service(
                RLT_PYTHON,
                "ur5e_real.adapters.rlinf_rlt.serve",
                [
                    "--run",
                    directory,
                    "--round",
                    round_dir,
                    "--feature-port",
                    feature_port,
                    "--feature-instance",
                    feature_instance,
                ],
                trial,
                round_dir,
                "heads",
            )
        )
        yield stack.enter_context(
            PolicyClient(run["contract"], port=port, timeout_s=run["config"]["real"]["rpc_timeout_s"])
        )


def parser():
    root = argparse.ArgumentParser(prog="ur5e-rlt", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create an immutable run recipe; no training or hardware")
    init.add_argument("run")
    init.add_argument("--checkpoint", default="joint5_sft_20260909_01:1000")
    init.add_argument("--config", type=Path, help="JSON configuration overrides")
    for name in ("status", "cache", "train-token", "train-bc", "train-ac", "round", "decide", "label"):
        command = sub.add_parser(name)
        command.add_argument("run")
        if name.startswith("train-"):
            command.add_argument("--steps", type=int, required=name != "train-token")
            command.add_argument("--device", default="cuda")
        if name == "cache":
            command.add_argument(
                "--limit", type=int, help="development-only partial cache; execute requires full cache"
            )
        if name == "train-ac":
            command.add_argument("--mode", choices=("warmup", "online"), required=True)
        if name == "round":
            group = command.add_mutually_exclusive_group(required=True)
            group.add_argument("--execute", action="store_true")
            group.add_argument("--dry-run", action="store_true")
            command.add_argument("--resume", help="unfinished round ID printed by status")
        if name == "decide":
            command.add_argument(
                "--stage", required=True, choices=("token", "bc", "reference", "actor_probe", "online", "complete")
            )
            command.add_argument("--checkpoint", type=Path)
            command.add_argument("--reason", required=True)
            command.add_argument("--evidence", required=True, type=Path)
        if name == "label":
            command.add_argument("--episode", type=Path, required=True)
            command.add_argument("--result", choices=("success", "failure", "timeout", "aborted"), required=True)
            command.add_argument("--reason", required=True)
    return root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    _enter_repository()
    try:
        if args.command == "init":
            from .pi05_operator import load_trial

            trial = load_trial(args.checkpoint)
            output = initialize(
                resolve_run(args.run),
                trial.dataset,
                trial.checkpoint,
                overrides=read_json(args.config) if args.config else None,
                lab_config=LAB_CONFIG,
            )
            print(f"[RLT RUN] {output}")
            return 0
        directory, run, state = load_run(args.run)
        if args.command == "status":
            print(json.dumps({"run": str(directory), "run_id": run["run_id"], **state}, ensure_ascii=False, indent=2))
            return 0
        target = NATIVE_PYTHON if args.command == "cache" else RLT_PYTHON if args.command.startswith("train-") else None
        if target and Path(sys.prefix).resolve() != target.parent.parent.resolve():
            if not target.is_file():
                raise FileNotFoundError(f"missing {target}; run scripts/setup_rlt_env.sh first")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPOSITORY / "src")
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            return subprocess.call([str(target), "-u", "-m", "ur5e_real.rlt_operator", *argv], env=env, cwd=REPOSITORY)
        # One local learner/cache/round at a time, including across different RLT runs.
        with ExitStack() as operation:
            operation.enter_context(lease(REPOSITORY / "logs" / "rlt" / ".device.lock"))
            operation.enter_context(lease(directory / ".run.lock"))
            append_json(
                directory / "invocations.jsonl",
                {
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    **implementation_identity(),
                },
            )
            if args.command in ("cache", "train-token", "train-bc", "train-ac") or (
                args.command == "round" and args.execute
            ):
                from .adapters.robotwin_pi05.telemetry import ResourceMonitor

                monitor = operation.enter_context(
                    ResourceMonitor(directory / "telemetry" / f"{args.command}_{uuid.uuid4().hex}")
                )
                monitor.phase(args.command)
            if args.command == "cache":
                from .adapters.rlinf_rlt.cache import build_cache

                result = build_cache(directory, limit=args.limit)
            elif args.command == "train-token":
                from .adapters.rlinf_rlt.train import train_token

                result = train_token(directory, steps=args.steps, device=args.device)
            elif args.command in ("train-bc", "train-ac"):
                from .adapters.rlinf_rlt.train import train_heads

                result = train_heads(
                    directory,
                    mode="bc" if args.command == "train-bc" else args.mode,
                    steps=args.steps,
                    device=args.device,
                )
            elif args.command == "decide":
                result = decide(directory, args.stage, args.checkpoint, args.reason, args.evidence)
            elif args.command == "label":
                from .adapters.rlinf_rlt.replay import label_episode, summarize

                episode = args.episode.resolve()
                if not episode.is_relative_to(directory / "rounds"):
                    raise ValueError("episode must belong to the selected run")
                result = label_episode(episode, args.result, reason=args.reason)
                summarize(episode.parent)
            elif args.command == "round":
                if args.dry_run:
                    print(
                        f"[DRY RUN] stage={state['stage']} serial episodes=4 K={run['config']['real']['action_steps']}; no device/model connection"
                    )
                    return 0
                if not sys.stdin.isatty():
                    raise ValueError("execute requires an interactive terminal for per-episode reset/start/result")
                from .adapters.rlinf_rlt.cache import load_cache
                from .adapters.rlinf_rlt.rounds import collect_round
                from .adapters.rlinf_rlt.environment import RealEnvironment
                from .adapters.rlinf_rlt.run import validate_base

                validate_base(run)
                if not load_cache(directory)["full_dataset"]:
                    raise ValueError("a development partial cache cannot run a physical round")
                verification = read_json(Path(run["checkpoint"]) / "ur5e_verification.json")
                if verification["status"] != "SFT_not_physical_validation":
                    raise ValueError("dummy base checkpoints are not eligible for physical execution")
                round_dir, spec = begin_round(directory, state["stage"], resume=args.resume)
                with round_policy(directory, run, round_dir) as policy:
                    result = collect_round(directory, round_dir, spec, policy, RealEnvironment(run))
            else:
                raise AssertionError(args.command)
            print(f"[RLT DONE] {result}", flush=True)
            return 0
    except (KeyboardInterrupt, EOFError):
        print(
            "\n[RLT PAUSED] inspect status; resume the unfinished round or learner from its last committed checkpoint"
        )
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[RLT ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
