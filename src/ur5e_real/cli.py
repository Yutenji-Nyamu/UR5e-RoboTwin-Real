from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ur5e-real", description="UR5e real-world tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="validate the environment and optional hardware")
    doctor.add_argument("--config", required=True)
    doctor.add_argument("--hardware", action="store_true")

    collect = sub.add_parser("collect", help="collect RTDE, gripper, and dual-camera data")
    collect.add_argument("--config", required=True)
    collect.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)

    convert = sub.add_parser("convert", help="convert raw sessions to RoboTwin-style HDF5")
    convert.add_argument("--config", required=True)
    convert.add_argument("--task", required=True)
    convert.add_argument("--task-config", default="simple")
    convert.add_argument("--output-root")

    preview = sub.add_parser("preview", help="inspect a converted HDF5 episode")
    preview.add_argument("hdf5_path")
    preview.add_argument("--camera", default="right_camera")
    preview.add_argument("--frames", type=int, default=4)
    preview.add_argument("--show", action="store_true")

    replay = sub.add_parser("replay", help="preview or execute a collected trajectory")
    replay.add_argument("--config", required=True)
    replay.add_argument("action_csv")
    replay.add_argument("--gripper-events")
    replay.add_argument("--row-stride", type=int, default=5)
    replay.add_argument("--max-segments", type=int)
    replay.add_argument("--execute", action="store_true", help="actually move the robot")

    process_act = sub.add_parser("process-act", help="convert real HDF5 episodes to RoboTwin ACT format")
    process_act.add_argument("run_root")
    process_act.add_argument("--robotwin-root", default=".third_party/RoboTwin")
    process_act.add_argument("--task", required=True)
    process_act.add_argument("--task-config", default="simple")
    process_act.add_argument("--episodes", type=int, required=True)

    infer_act = sub.add_parser("infer-act", help="run a RoboTwin ACT checkpoint on the real robot")
    infer_act.add_argument("--config", required=True)
    infer_act.add_argument("--robotwin-root", default=".third_party/RoboTwin")
    infer_act.add_argument("--task", required=True)
    infer_act.add_argument("--task-config", default="simple")
    infer_act.add_argument("--episodes", type=int, required=True)
    infer_act.add_argument("--checkpoint", default="policy_best.ckpt")
    infer_act.add_argument("--no-gripper", action="store_true")
    infer_act.add_argument("--execute", action="store_true", help="actually stream commands to the robot")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "doctor":
        from .doctor import print_checks, run_doctor

        return 0 if print_checks(run_doctor(load_config(args.config), hardware=args.hardware)) else 1

    if args.command == "collect":
        from .collection.session import run_collection

        run_collection(load_config(args.config), preview=args.preview, save_video=args.save_video)
        return 0

    if args.command == "convert":
        from .data.convert_hdf5 import convert_raw_sessions

        cfg = load_config(args.config)
        output_root = Path(args.output_root).resolve() if args.output_root else cfg.collection.data_root / "converted"
        result = convert_raw_sessions(
            cfg.collection.data_root / "raw" / "action",
            cfg.collection.data_root / "raw" / "camera",
            output_root,
            args.task,
            args.task_config,
        )
        print(result)
        return 0

    if args.command == "preview":
        from .data.preview_hdf5 import preview_file

        preview_file(Path(args.hdf5_path), args.camera, args.frames, args.show)
        return 0

    if args.command == "replay":
        from .replay import ReplayConfig, run_replay

        cfg = load_config(args.config)
        replay_cfg = ReplayConfig(
            action_csv=Path(args.action_csv),
            gripper_events=Path(args.gripper_events) if args.gripper_events else None,
            robot_host=cfg.robot.host,
            robot_port=cfg.robot.script_port,
            socket_timeout_s=cfg.robot.socket_timeout_s,
            serial_port=cfg.gripper.port,
            serial_baudrate=cfg.gripper.baudrate,
            serial_timeout_s=cfg.gripper.timeout_s,
            row_stride=args.row_stride,
            max_segments=args.max_segments,
            execute=args.execute,
        )
        run_replay(replay_cfg)
        return 0

    if args.command == "process-act":
        from .adapters.robotwin_act.process_data import process_run

        process_run(
            Path(args.run_root).resolve(),
            Path(args.robotwin_root).resolve(),
            args.task,
            args.task_config,
            args.episodes,
        )
        return 0

    if args.command == "infer-act":
        from .adapters.robotwin_act.infer_act import InferenceConfig, run_inference

        run_inference(
            load_config(args.config),
            Path(args.robotwin_root).resolve(),
            InferenceConfig(args.task, args.task_config, args.episodes, args.checkpoint),
            execute=args.execute,
            enable_gripper=not args.no_gripper,
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
