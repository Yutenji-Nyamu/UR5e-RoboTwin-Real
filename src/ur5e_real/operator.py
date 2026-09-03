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
