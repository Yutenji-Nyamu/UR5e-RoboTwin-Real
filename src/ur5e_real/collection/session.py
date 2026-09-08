from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import LabConfig
from ..data.schema import RAW_SCHEMA_VERSION
from ..data.session_manifest import write_manifest
from ..hardware.gripper import GripperSerial
from ..hardware.realsense import DualColorCamera
from ..hardware.rtde import RtdeOutputConfig, RtdeStateClient, RtdeStateCsvWriter
from ..hardware.urscript import start_freedrive, stop_freedrive
from .terminal import TerminalKeyPoller


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _code_commit() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def run_collection(
    cfg: LabConfig,
    *,
    task: str,
    initial_gripper_state: str,
    note: str | None = None,
    preview: bool | None = None,
    save_video: bool | None = None,
) -> Path:
    import cv2

    task = task.strip()
    if not task:
        raise ValueError("task must not be empty")
    if initial_gripper_state not in {"open", "closed"}:
        raise ValueError("initial_gripper_state must be 'open' or 'closed'")

    show_preview = cfg.collection.preview if preview is None else preview
    write_video = cfg.collection.save_video if save_video is None else save_video
    run_id = _run_id()
    raw_root = cfg.collection.data_root / "raw"
    action_dir = raw_root / "action"
    camera_dir = raw_root / "camera" / f"cam_dual_{run_id}"
    head_dir = camera_dir / "head"
    wrist_dir = camera_dir / "wrist"
    rtde_path = action_dir / f"rtde_tcp_gripper_{run_id}.csv"
    events_path = action_dir / f"gripper_events_{run_id}.csv"
    sync_path = action_dir / f"sync_action_cam_{run_id}.csv"
    manifest_path = action_dir / f"session_{run_id}.json"
    for path in (camera_dir, rtde_path, events_path, sync_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing session product: {path}")
    action_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=False)
    head_dir.mkdir()
    wrist_dir.mkdir()
    manifest = {
        "schema_version": RAW_SCHEMA_VERSION,
        "run_id": run_id,
        "task": task,
        "note": note or None,
        "started_at": None,
        "finished_at": None,
        "duration_s": None,
        "recording_status": "initializing",
        "stop_reason": None,
        "outcome": "unreviewed",
        "reviews": [],
        "code_commit": _code_commit(),
        "robot": asdict(cfg.robot),
        "gripper": asdict(cfg.gripper),
        "cameras": asdict(cfg.cameras),
        "state_recording": {
            "representations": ["tcp", "joint"],
            "rtde_fields": list(RtdeStateClient.OUTPUT_FIELDS),
            "csv_columns": RtdeStateCsvWriter.COLUMNS,
            "joint_source": "actual_q",
            "joint_velocity_source": "actual_qd",
            "tcp_source": "actual_TCP_pose",
            "joint_order": ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"],
            "joint_position_unit": "rad",
            "joint_velocity_unit": "rad/s",
            "tcp_pose_units": ["m", "m", "m", "rad", "rad", "rad"],
            "tcp_pose_frame": "robot_base",
            "tcp_orientation": "rotation_vector",
            "sample_relation": "same_rtde_packet",
            "host_receive_time_clock": "unix_seconds_not_camera_exposure",
            "gripper_state_semantics": "commanded_open_0_closed_1_not_measured_width",
            "tcp_offset_source": "tcp_offset_flange_to_tcp",
        },
        "initial_robot_state": None,
        "collection": {
            "data_root": str(cfg.collection.data_root),
            "enable_freedrive_on_start": cfg.collection.enable_freedrive_on_start,
            "initial_gripper_state": initial_gripper_state,
            "preview": show_preview,
            "save_video": write_video,
        },
        "paths": {
            "rtde": str(rtde_path),
            "gripper_events": str(events_path),
            "sync": str(sync_path),
            "camera": str(camera_dir),
            "head_frames": str(head_dir),
            "wrist_frames": str(wrist_dir),
        },
        "counts": {
            "rtde_samples": 0,
            "gripper_events": 0,
            "frame_pairs": 0,
        },
    }
    if write_video:
        manifest["paths"]["head_video"] = str(camera_dir / "head.mp4")
        manifest["paths"]["wrist_video"] = str(camera_dir / "wrist.mp4")
    write_manifest(manifest_path, manifest)

    cameras = DualColorCamera(
        cfg.cameras.head_serial,
        cfg.cameras.wrist_serial,
        cfg.cameras.width,
        cfg.cameras.height,
        cfg.cameras.fps,
        cfg.cameras.warmup_frames,
    )
    rtde = RtdeStateClient(RtdeOutputConfig(cfg.robot.host, cfg.robot.rtde_port, cfg.robot.rtde_frequency_hz))
    gripper: GripperSerial | None = None
    rtde_writer: RtdeStateCsvWriter | None = None
    events_handle: Any = None
    sync_handle: Any = None
    head_video: Any = None
    wrist_video: Any = None
    freedrive_started = False
    started_monotonic: float | None = None
    rtde_sample_count = 0
    event_counter = 0
    frame_index = 0
    previous_controller_time: float | None = None
    last_open_time: float | None = None
    last_frame_time: float | None = None

    try:
        cameras.start()
        rtde.connect()
        initial_state = rtde.receive_state()
        if initial_state is None:
            raise RuntimeError("RTDE connection closed before the first joint/TCP state")
        manifest["initial_robot_state"] = asdict(initial_state)
        previous_controller_time = initial_state.controller_time_s
        gripper = GripperSerial(cfg.gripper.port, cfg.gripper.baudrate, cfg.gripper.timeout_s)
        rtde_writer = RtdeStateCsvWriter(rtde_path)

        events_handle = events_path.open("x", newline="", encoding="utf-8")
        events_writer = csv.writer(events_handle)
        events_writer.writerow(["controller_time_s", "event", "gripper_state"])
        events_handle.flush()

        sync_handle = sync_path.open("x", newline="", encoding="utf-8")
        sync_writer = csv.writer(sync_handle)
        sync_writer.writerow(["controller_time_s", "frame_idx", "head_image", "wrist_image"])
        sync_handle.flush()

        if write_video:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            size = (cfg.cameras.width, cfg.cameras.height)
            head_video = cv2.VideoWriter(str(camera_dir / "head.mp4"), codec, cfg.cameras.save_hz, size)
            wrist_video = cv2.VideoWriter(str(camera_dir / "wrist.mp4"), codec, cfg.cameras.save_hz, size)
            if not head_video.isOpened() or not wrist_video.isOpened():
                raise RuntimeError("failed to initialize MP4 writers")

        if cfg.collection.enable_freedrive_on_start:
            start_freedrive(
                cfg.robot.host,
                cfg.robot.script_port,
                cfg.robot.socket_timeout_s,
            )
            freedrive_started = True

        gripper_state = 1 if initial_gripper_state == "closed" else 0
        manifest["started_at"] = datetime.now().astimezone().isoformat()
        manifest["recording_status"] = "recording"
        write_manifest(manifest_path, manifest)
        started_monotonic = time.monotonic()
        next_save = time.monotonic()
        print(f"[RUN] {run_id}")
        print(f"[STATE] schema={RAW_SCHEMA_VERSION} actual_q[6] + actual_qd[6] + TCP[6]; same RTDE packet")
        print(f"[READY] recording active; freedrive={'on' if freedrive_started else 'off'}")
        print("Keys: c=close, o=open, q=quit; Ctrl+C also stops.")
        print("Keep recording for at least 1 second after the final open, until release is complete.")

        with TerminalKeyPoller() as keys:
            if not keys.enabled:
                print("[WARN] stdin is not an interactive terminal; only Ctrl+C can stop collection")
            while True:
                sample = rtde.receive_state()
                if sample is None:
                    raise RuntimeError("RTDE connection closed")
                controller_time = sample.controller_time_s
                if previous_controller_time is not None and controller_time <= previous_controller_time:
                    raise RuntimeError("RTDE controller time did not advance; refusing an invalid timeline")
                previous_controller_time = controller_time
                key = keys.poll()
                if key == "q":
                    break
                if key in {"c", "o"}:
                    if key == "c":
                        gripper.close()
                        event = "close"
                        gripper_state = 1
                    else:
                        gripper.open()
                        event = "open"
                        gripper_state = 0
                        last_open_time = controller_time
                    event_counter += 1
                    events_writer.writerow([controller_time, event, gripper_state])
                    events_handle.flush()

                # Preserve robot states even if a camera pair is unavailable.
                rtde_writer.write(sample, gripper_state, event_counter)
                rtde_sample_count += 1
                pair = cameras.read()
                if pair is None:
                    continue

                if show_preview:
                    cv2.imshow("head", pair.head)
                    cv2.imshow("wrist", pair.wrist)
                    cv2.waitKey(1)

                now = time.monotonic()
                if now >= next_save:
                    next_frame_index = frame_index + 1
                    head_name = f"frame_{next_frame_index:05d}.png"
                    wrist_name = f"frame_{next_frame_index:05d}.png"
                    if not cv2.imwrite(str(head_dir / head_name), pair.head):
                        raise RuntimeError("failed to write head camera frame")
                    if not cv2.imwrite(str(wrist_dir / wrist_name), pair.wrist):
                        raise RuntimeError("failed to write wrist camera frame")
                    if head_video is not None:
                        head_video.write(pair.head)
                        wrist_video.write(pair.wrist)
                    sync_writer.writerow([controller_time, next_frame_index, head_name, wrist_name])
                    sync_handle.flush()
                    frame_index = next_frame_index
                    last_frame_time = controller_time
                    next_save += 1.0 / cfg.cameras.save_hz
        manifest["recording_status"] = "completed"
        manifest["stop_reason"] = "user_quit"
    except KeyboardInterrupt:
        manifest["recording_status"] = "interrupted"
        manifest["stop_reason"] = "keyboard_interrupt"
        print("\n[STOP] interrupted")
    except Exception as exc:
        manifest["recording_status"] = "failed"
        manifest["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if freedrive_started:
            try:
                stop_freedrive(cfg.robot.host, cfg.robot.script_port, cfg.robot.socket_timeout_s)
            except Exception as exc:
                print(f"[WARN] failed to stop freedrive: {exc!r}")
        if gripper is not None:
            gripper.shutdown()
        rtde.close()
        cameras.stop()
        if rtde_writer is not None:
            rtde_writer.close()
        for handle in (events_handle, sync_handle):
            if handle is not None:
                handle.close()
        for writer in (head_video, wrist_video):
            if writer is not None:
                writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        if started_monotonic is not None:
            manifest["duration_s"] = round(time.monotonic() - started_monotonic, 3)
        manifest["counts"] = {
            "rtde_samples": rtde_sample_count,
            "gripper_events": event_counter,
            "frame_pairs": frame_index,
        }
        release_tail = (
            round(last_frame_time - last_open_time, 6)
            if last_frame_time is not None and last_open_time is not None
            else None
        )
        manifest["quality"] = {
            "last_open_controller_time_s": last_open_time,
            "last_frame_controller_time_s": last_frame_time,
            "last_open_to_last_frame_s": release_tail,
            "recommended_release_tail_s": 1.0,
            "release_tail_complete": (
                None if last_open_time is None else release_tail is not None and release_tail >= 1.0
            ),
        }
        write_manifest(manifest_path, manifest)
        if manifest["quality"]["release_tail_complete"] is False:
            print("[WARN] less than 1 second of images after final open; check release before marking success")

    print(f"[SAVED] {manifest_path}")
    return manifest_path
