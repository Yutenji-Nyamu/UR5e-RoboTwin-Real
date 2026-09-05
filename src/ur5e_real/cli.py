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

    prepare = sub.add_parser("prepare", help="move to the configured home TCP and open the gripper")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--execute", action="store_true", help="actually move the robot and open the gripper")

    collect = sub.add_parser("collect", help="collect RTDE, gripper, and dual-camera data")
    collect.add_argument("--config", required=True)
    collect.add_argument("--task", required=True, help="short task name stored in the session manifest")
    collect.add_argument(
        "--initial-gripper",
        required=True,
        choices=("open", "closed"),
        help="physical gripper state at recording start; does not command the gripper",
    )
    collect.add_argument("--note", help="optional setup or variation note")
    collect.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)

    sessions = sub.add_parser("sessions", help="list recent raw recording sessions")
    sessions.add_argument("--config", required=True)
    sessions.add_argument("--limit", type=int, default=20, help="newest sessions to show; 0 means all")

    review = sub.add_parser("review", help="append a success/failure review to a session manifest")
    review.add_argument("session_json")
    review.add_argument("--result", required=True, choices=("success", "failure", "aborted"))
    review.add_argument("--note")

    convert = sub.add_parser("convert", help="convert raw sessions to RoboTwin-style HDF5")
    convert.add_argument("--config", required=True)
    convert.add_argument("--task", required=True)
    convert.add_argument("--task-config", default="simple")
    convert.add_argument("--output-root")
    convert.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="also convert legacy or non-success sessions",
    )

    preview = sub.add_parser("preview", help="inspect a converted HDF5 episode")
    preview.add_argument("hdf5_path")
    preview.add_argument("--camera", default="right_camera")
    preview.add_argument("--frames", type=int, default=4)
    preview.add_argument("--show", action="store_true")

    replay = sub.add_parser("replay", help="preview or execute a collected trajectory")
    replay.add_argument("--config", required=True)
    replay.add_argument("source", help="session JSON, or an action CSV for legacy data")
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

    process_dp = sub.add_parser("process-dp", help="convert real HDF5 episodes to RoboTwin DP Zarr")
    process_dp.add_argument("run_root")
    process_dp.add_argument("--task", required=True)
    process_dp.add_argument("--task-config", default="simple")
    process_dp.add_argument("--episodes", type=int, required=True)
    process_dp.add_argument("--output")
    process_dp.add_argument("--overwrite", action="store_true")
    process_dp.add_argument("--trim-static-edges", action="store_true")
    process_dp.add_argument("--motion-threshold-mm", type=float, default=2.0)
    process_dp.add_argument("--motion-threshold-deg", type=float, default=1.0)
    process_dp.add_argument("--lead-in-steps", type=int, default=3)
    process_dp.add_argument("--tail-steps", type=int, default=3)

    train_dp = sub.add_parser("train-dp", help="train RoboTwin DP on a converted real dataset")
    train_dp.add_argument("zarr_path")
    train_dp.add_argument("--robotwin-root", default=".third_party/RoboTwin")
    train_dp.add_argument("--task", required=True)
    train_dp.add_argument("--task-config", default="simple")
    train_dp.add_argument("--episodes", type=int, required=True)
    train_dp.add_argument("--seed", type=int, default=0)
    train_dp.add_argument("--gpu", default="0")
    train_dp.add_argument("--debug", action="store_true")
    train_dp.add_argument("--batch-size", type=int)
    train_dp.add_argument("--val-ratio", type=float)
    train_dp.add_argument("--output-dir")

    infer_dp = sub.add_parser("infer-dp", help="run RoboTwin DP offline, shadow, or through RTDE servoJ")
    infer_dp.add_argument("--robotwin-root", default=".third_party/RoboTwin")
    infer_dp.add_argument("--checkpoint", required=True)
    dp_mode = infer_dp.add_mutually_exclusive_group(required=True)
    dp_mode.add_argument("--episode", help="offline HDF5 episode; never connects hardware")
    dp_mode.add_argument("--shadow", action="store_true", help="live inference without sending commands")
    dp_mode.add_argument("--execute", action="store_true", help="execute through the selected motion backend")
    infer_dp.add_argument("--config", help="lab config required for shadow or execute")
    infer_dp.add_argument("--index", type=int, default=2)
    infer_dp.add_argument("--output")
    infer_dp.add_argument("--chunks", type=int, default=1, help="live chunks; 0 means until Ctrl+C")
    infer_dp.add_argument("--gpu", default="0")
    infer_dp.add_argument("--no-gripper", action="store_true")
    infer_dp.add_argument("--backend", choices=("socket", "rtde"), default="socket")
    infer_dp.add_argument("--smooth-alpha", type=float, default=0.7)
    infer_dp.add_argument("--max-linear-speed", type=float, default=0.20)
    infer_dp.add_argument("--diffusion-steps", type=int, default=100)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "doctor":
        from .doctor import print_checks, run_doctor

        return 0 if print_checks(run_doctor(load_config(args.config), hardware=args.hardware)) else 1

    if args.command == "prepare":
        from .control.prepare import run_prepare

        run_prepare(load_config(args.config), execute=args.execute)
        return 0

    if args.command == "collect":
        from .collection.session import run_collection

        run_collection(
            load_config(args.config),
            task=args.task,
            initial_gripper_state=args.initial_gripper,
            note=args.note,
            preview=args.preview,
            save_video=args.save_video,
        )
        return 0

    if args.command == "sessions":
        from .data.session_manifest import print_session_summaries

        cfg = load_config(args.config)
        print_session_summaries(cfg.collection.data_root / "raw" / "action", args.limit)
        return 0

    if args.command == "review":
        from .data.session_manifest import review_session

        path = Path(args.session_json).expanduser().resolve()
        review = review_session(path, args.result, args.note)
        print(f"[REVIEWED] {path}: {review['result']}")
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
            include_unreviewed=args.include_unreviewed,
        )
        print(result)
        return 0

    if args.command == "preview":
        from .data.preview_hdf5 import preview_file

        preview_file(Path(args.hdf5_path), args.camera, args.frames, args.show)
        return 0

    if args.command == "replay":
        from .replay import ReplayConfig, resolve_replay_paths, run_replay

        cfg = load_config(args.config)
        action_csv, gripper_events = resolve_replay_paths(
            Path(args.source).expanduser().resolve(),
            Path(args.gripper_events).expanduser().resolve() if args.gripper_events else None,
        )
        replay_cfg = ReplayConfig(
            action_csv=action_csv,
            gripper_events=gripper_events,
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

    if args.command == "process-dp":
        from .adapters.robotwin_dp.process_data import process_run

        process_run(
            Path(args.run_root),
            args.task,
            args.task_config,
            args.episodes,
            output_path=Path(args.output) if args.output else None,
            overwrite=args.overwrite,
            trim_static_edges=args.trim_static_edges,
            motion_threshold_mm=args.motion_threshold_mm,
            motion_threshold_deg=args.motion_threshold_deg,
            lead_in_steps=args.lead_in_steps,
            tail_steps=args.tail_steps,
        )
        return 0

    if args.command == "train-dp":
        from .adapters.robotwin_dp.train_dp import TrainConfig, run_training

        run_training(
            Path(args.robotwin_root),
            Path(args.zarr_path),
            TrainConfig(
                task_name=args.task,
                task_config=args.task_config,
                episode_count=args.episodes,
                seed=args.seed,
                gpu=args.gpu,
                debug=args.debug,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
            ),
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        return 0

    if args.command == "infer-dp":
        from .adapters.robotwin_dp.infer_dp import InferenceConfig as DPInferenceConfig
        from .adapters.robotwin_dp.infer_dp import run_execute, run_offline, run_shadow

        inference = DPInferenceConfig(
            chunks=args.chunks,
            gpu=args.gpu,
            enable_gripper=not args.no_gripper,
            backend=args.backend,
            smoothing_alpha=args.smooth_alpha,
            max_linear_velocity=args.max_linear_speed,
            diffusion_steps=args.diffusion_steps,
        )
        robotwin_root = Path(args.robotwin_root).expanduser().resolve()
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if args.episode:
            run_offline(
                robotwin_root,
                checkpoint,
                Path(args.episode).expanduser().resolve(),
                inference,
                index=args.index,
                output=Path(args.output) if args.output else None,
            )
            return 0
        if not args.config:
            raise SystemExit("--config is required with --shadow or --execute")
        lab = load_config(args.config)
        if args.shadow:
            run_shadow(lab, robotwin_root, checkpoint, inference)
        else:
            run_execute(lab, robotwin_root, checkpoint, inference)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
