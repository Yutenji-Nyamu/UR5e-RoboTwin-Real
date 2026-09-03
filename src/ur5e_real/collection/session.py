from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import LabConfig
from ..data.session_manifest import write_manifest
from ..hardware.gripper import GripperSerial
from ..hardware.realsense import DualColorCamera
from ..hardware.rtde import RtdeCsvWriter, RtdeOutputConfig, RtdeTcpClient
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
    for directory in (action_dir, head_dir, wrist_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rtde_path = action_dir / f"rtde_tcp_gripper_{run_id}.csv"
    events_path = action_dir / f"gripper_events_{run_id}.csv"
    sync_path = action_dir / f"sync_action_cam_{run_id}.csv"
    manifest_path = action_dir / f"session_{run_id}.json"
    manifest = {
        "schema_version": 2,
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
    rtde = RtdeTcpClient(
        RtdeOutputConfig(cfg.robot.host, cfg.robot.rtde_port, cfg.robot.rtde_frequency_hz)
    )
    gripper: GripperSerial | None = None
    rtde_writer: RtdeCsvWriter | None = None
    events_handle: Any = None
    sync_handle: Any = None
    head_video: Any = None
    wrist_video: Any = None
    freedrive_started = False
    started_monotonic: float | None = None
    rtde_sample_count = 0
    event_counter = 0
    frame_index = 0

    try:
        cameras.start()
        rtde.connect()
        gripper = GripperSerial(cfg.gripper.port, cfg.gripper.baudrate, cfg.gripper.timeout_s)
        rtde_writer = RtdeCsvWriter(rtde_path)

        events_handle = events_path.open("w", newline="", encoding="utf-8")
        events_writer = csv.writer(events_handle)
        events_writer.writerow(["controller_time_s", "event", "gripper_state"])
        events_handle.flush()

        sync_handle = sync_path.open("w", newline="", encoding="utf-8")
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
        print("Keys: c=close, o=open, q=quit; Ctrl+C also stops.")

        with TerminalKeyPoller() as keys:
            if not keys.enabled:
                print("[WARN] stdin is not an interactive terminal; only Ctrl+C can stop collection")
            while True:
                sample = rtde.receive()
                if sample is None:
                    raise RuntimeError("RTDE connection closed")
                controller_time, pose = sample
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
                    event_counter += 1
                    events_writer.writerow([controller_time, event, gripper_state])
                    events_handle.flush()

                pair = cameras.read()
                if pair is None:
                    continue
                rtde_writer.write(controller_time, pose, gripper_state, event_counter)
                rtde_sample_count += 1

                if show_preview:
                    cv2.imshow("head", pair.head)
                    cv2.imshow("wrist", pair.wrist)
                    cv2.waitKey(1)

                now = time.monotonic()
                if now >= next_save:
                    frame_index += 1
                    head_name = f"frame_{frame_index:05d}.png"
                    wrist_name = f"frame_{frame_index:05d}.png"
                    if not cv2.imwrite(str(head_dir / head_name), pair.head):
                        raise RuntimeError("failed to write head camera frame")
                    if not cv2.imwrite(str(wrist_dir / wrist_name), pair.wrist):
                        raise RuntimeError("failed to write wrist camera frame")
                    if head_video is not None:
                        head_video.write(pair.head)
                        wrist_video.write(pair.wrist)
                    sync_writer.writerow([controller_time, frame_index, head_name, wrist_name])
                    sync_handle.flush()
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
        write_manifest(manifest_path, manifest)

    print(f"[SAVED] {manifest_path}")
    return manifest_path
