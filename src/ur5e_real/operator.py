from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config

REPOSITORY = Path(__file__).resolve().parents[2]
LAB_CONFIG = REPOSITORY / "configs" / "lab.yaml"


def _enter_repository() -> None:
    os.chdir(REPOSITORY)
    if not LAB_CONFIG.is_file():
        raise FileNotFoundError(f"local configuration not found: {LAB_CONFIG}")


def _initialize(flow: str, *, execute: bool) -> int:
    from .control.prepare import run_prepare
    from .doctor import print_checks, run_doctor

    _enter_repository()
    cfg = load_config(LAB_CONFIG)
    print(f"[{flow.upper()} INIT] environment and repository selected")
    if not print_checks(run_doctor(cfg, hardware=True)):
        return 1
    run_prepare(cfg, execute=execute)
    return 0


def _init_command(flow: str) -> int:
    parser = argparse.ArgumentParser(
        prog=f"ur5e-{flow}-init",
        description=f"check hardware, move to home, and open the gripper before {flow}",
    )
    parser.add_argument("--dry-run", action="store_true", help="check hardware and print the plan without motion")
    args = parser.parse_args()
    try:
        return _initialize(flow, execute=not args.dry_run)
    except RuntimeError as exc:
        print(f"[BLOCKED] {exc}")
        return 2


def collect_init() -> int:
    return _init_command("collect")


def replay_init() -> int:
    return _init_command("replay")


def policy_init() -> int:
    return _init_command("policy")


def infer_init() -> int:
    return _init_command("infer")


def collect() -> int:
    from .collection.session import run_collection
    from .data.session_manifest import review_session

    parser = argparse.ArgumentParser(prog="ur5e-collect", description="teleoperate and record one trajectory")
    parser.add_argument("task", help="task name, for example pick_block_bowl")
    parser.add_argument("--note")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    _enter_repository()
    manifest = run_collection(
        load_config(LAB_CONFIG),
        task=args.task,
        initial_gripper_state="open",
        note=args.note,
        preview=args.preview,
        save_video=args.save_video,
    )
    choices = {"s": "success", "f": "failure", "a": "aborted"}
    while True:
        try:
            answer = input("Result [s=success, f=failure, a=aborted, Enter=later]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[UNREVIEWED] result can be added later")
            break
        if not answer:
            break
        if answer in choices:
            review_session(manifest, choices[answer])
            print(f"[REVIEWED] {choices[answer]}")
            break
        print("Please enter s, f, a, or Enter.")
    return 0


def _session_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    action_dir = load_config(LAB_CONFIG).collection.data_root / "raw" / "action"
    filename = value if value.startswith("session_") else f"session_{value}"
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    return action_dir / filename


def resolve_dp_checkpoint(value: str, data_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.parent != Path("."):
        raise FileNotFoundError(candidate)

    run_hint = None
    checkpoint_name = value
    if ":" in value:
        run_hint, checkpoint_name = value.rsplit(":", 1)
    filename = checkpoint_name if checkpoint_name.endswith(".ckpt") else f"{checkpoint_name}.ckpt"
    checkpoint_root = data_root / "checkpoints" / "dp"
    matches = sorted(path.resolve() for path in checkpoint_root.rglob(filename) if path.is_file())
    if run_hint:
        matches = [path for path in matches if run_hint in str(path)]
    if not matches:
        qualifier = f" in run matching {run_hint}" if run_hint else ""
        raise FileNotFoundError(f"no {filename}{qualifier} under {checkpoint_root}")
    if len(matches) > 1:
        paths = "\n".join(f"  {path}" for path in matches)
        raise RuntimeError(f"checkpoint name {filename} is ambiguous; pass a full path:\n{paths}")
    return matches[0]


def infer() -> int:
    from .adapters.robotwin_dp.infer_dp import InferenceConfig, run_execute, run_shadow

    parser = argparse.ArgumentParser(
        prog="ur5e-infer",
        description="run one RoboTwin DP checkpoint with the local lab configuration",
    )
    parser.add_argument(
        "checkpoint",
        help="path, unique epoch such as 600, or timestamp-qualified ID such as 20260905_120000:600",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shadow", action="store_true", help="predict from live cameras without commands")
    mode.add_argument("--execute", action="store_true", help="execute through the selected motion backend")
    parser.add_argument(
        "--chunks",
        type=int,
        default=50,
        help="six-action chunks to run (default: 50 = RoboTwin's 300-action episode; 0 until Ctrl+C)",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--backend", choices=("socket", "rtde"), default="socket")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="socket target EMA; 1 disables smoothing (default: 0.7)",
    )
    args = parser.parse_args()
    if args.chunks < 0:
        parser.error("--chunks must be non-negative")

    _enter_repository()
    cfg = load_config(LAB_CONFIG)
    checkpoint = resolve_dp_checkpoint(args.checkpoint, cfg.collection.data_root)
    inference = InferenceConfig(
        chunks=args.chunks,
        gpu=args.gpu,
        enable_gripper=not args.no_gripper,
        backend=args.backend,
        smoothing_alpha=args.smooth_alpha,
    )
    mode_name = "shadow" if args.shadow else "execute"
    backend = "read-only" if args.shadow else args.backend
    print(
        f"[INFER] mode={mode_name} backend={backend} chunks={args.chunks} "
        f"checkpoint={checkpoint}"
    )
    robotwin_root = REPOSITORY / ".third_party" / "RoboTwin"
    if args.shadow:
        run_shadow(cfg, robotwin_root, checkpoint, inference)
    else:
        run_execute(cfg, robotwin_root, checkpoint, inference)
    return 0


def replay() -> int:
    from .replay import ReplayConfig, resolve_replay_paths, run_replay

    parser = argparse.ArgumentParser(prog="ur5e-replay", description="preview or replay one recorded trajectory")
    parser.add_argument("session", help="run ID or session JSON path")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    _enter_repository()
    cfg = load_config(LAB_CONFIG)
    session = _session_path(args.session)
    action_csv, events = resolve_replay_paths(session)
    run_replay(
        ReplayConfig(
            action_csv=action_csv,
            gripper_events=events,
            robot_host=cfg.robot.host,
            robot_port=cfg.robot.script_port,
            socket_timeout_s=cfg.robot.socket_timeout_s,
            serial_port=cfg.gripper.port,
            serial_baudrate=cfg.gripper.baudrate,
            serial_timeout_s=cfg.gripper.timeout_s,
            max_segments=args.max_segments,
            execute=args.execute,
        )
    )
    return 0
