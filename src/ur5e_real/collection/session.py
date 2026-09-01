from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import LabConfig
from ..hardware.gripper import GripperSerial
from ..hardware.realsense import DualColorCamera
from ..hardware.rtde import RtdeCsvWriter, RtdeOutputConfig, RtdeTcpClient
from ..hardware.urscript import start_freedrive, stop_freedrive
from .terminal import TerminalKeyPoller


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_collection(cfg: LabConfig, *, preview: bool | None = None, save_video: bool | None = None) -> Path:
    import cv2

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
        "schema_version": 1,
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "robot": asdict(cfg.robot),
        "gripper": asdict(cfg.gripper),
        "cameras": asdict(cfg.cameras),
        "collection": {
            "data_root": str(cfg.collection.data_root),
            "enable_freedrive_on_start": cfg.collection.enable_freedrive_on_start,
            "preview": show_preview,
            "save_video": write_video,
        },
        "paths": {
            "rtde": str(rtde_path),
            "gripper_events": str(events_path),
            "sync": str(sync_path),
            "camera": str(camera_dir),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cameras = DualColorCamera(
        cfg.cameras.head_serial,
        cfg.cameras.wrist_serial,
        cfg.cameras.width,
        cfg.cameras.height,
        cfg.cameras.fps,
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

        gripper_state = 0
        event_counter = 0
        frame_index = 0
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
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
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

    print(f"[SAVED] {manifest_path}")
    return manifest_path
