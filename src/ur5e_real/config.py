from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class RobotConfig:
    host: str
    script_port: int = 30001
    rtde_port: int = 30004
    rtde_frequency_hz: float = 10.0
    socket_timeout_s: float = 10.0
    home_tcp_pose: tuple[float, ...] | None = None


@dataclass(frozen=True)
class GripperConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout_s: float = 1.0


@dataclass(frozen=True)
class CameraConfig:
    head_serial: str
    wrist_serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    save_hz: float = 10.0
    warmup_frames: int = 60


@dataclass(frozen=True)
class CollectionConfig:
    data_root: Path
    enable_freedrive_on_start: bool = True
    preview: bool = False
    save_video: bool = False


@dataclass(frozen=True)
class ServoJConfig:
    frequency_hz: int
    config_xml: Path
    mode: int = 2
    connect_timeout_s: float = 15.0


@dataclass(frozen=True)
class LabConfig:
    robot: RobotConfig
    gripper: GripperConfig
    cameras: CameraConfig
    collection: CollectionConfig
    servoj: ServoJConfig
    source_path: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> LabConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = _mapping(raw, "configuration")
    base = source.parent

    robot = _mapping(root.get("robot"), "robot")
    gripper = _mapping(root.get("gripper", {}), "gripper")
    cameras = _mapping(root.get("cameras"), "cameras")
    collection = _mapping(root.get("collection", {}), "collection")
    servoj = _mapping(root.get("servoj", {}), "servoj")

    host = str(robot.get("host", "")).strip()
    head_serial = str(cameras.get("head_serial", "")).strip()
    wrist_serial = str(cameras.get("wrist_serial", "")).strip()
    missing = [
        name
        for name, value in (
            ("robot.host", host),
            ("cameras.head_serial", head_serial),
            ("cameras.wrist_serial", wrist_serial),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"missing required configuration: {', '.join(missing)}")

    cfg = LabConfig(
        robot=RobotConfig(
            host=host,
            script_port=int(robot.get("script_port", 30001)),
            rtde_port=int(robot.get("rtde_port", 30004)),
            rtde_frequency_hz=float(robot.get("rtde_frequency_hz", 10.0)),
            socket_timeout_s=float(robot.get("socket_timeout_s", 10.0)),
            home_tcp_pose=(
                tuple(float(value) for value in robot["home_tcp_pose"])
                if robot.get("home_tcp_pose") is not None
                else None
            ),
        ),
        gripper=GripperConfig(
            port=str(gripper.get("port", "/dev/ttyUSB0")),
            baudrate=int(gripper.get("baudrate", 9600)),
            timeout_s=float(gripper.get("timeout_s", 1.0)),
        ),
        cameras=CameraConfig(
            head_serial=head_serial,
            wrist_serial=wrist_serial,
            width=int(cameras.get("width", 640)),
            height=int(cameras.get("height", 480)),
            fps=int(cameras.get("fps", 30)),
            save_hz=float(cameras.get("save_hz", 10.0)),
            warmup_frames=int(cameras.get("warmup_frames", 60)),
        ),
        collection=CollectionConfig(
            data_root=_resolve(base, str(collection.get("data_root", "../data"))),
            enable_freedrive_on_start=bool(collection.get("enable_freedrive_on_start", True)),
            preview=bool(collection.get("preview", False)),
            save_video=bool(collection.get("save_video", False)),
        ),
        servoj=ServoJConfig(
            frequency_hz=int(servoj.get("frequency_hz", 500)),
            config_xml=_resolve(base, str(servoj.get("config_xml", "../robot_programs/control_loop_configuration.xml"))),
            mode=int(servoj.get("mode", 2)),
            connect_timeout_s=float(servoj.get("connect_timeout_s", 15.0)),
        ),
        source_path=source,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: LabConfig) -> None:
    if cfg.robot.host in {"robot-ip-or-hostname", "CHANGEME"}:
        raise ValueError("robot.host still contains the example placeholder")
    if cfg.cameras.head_serial == cfg.cameras.wrist_serial:
        raise ValueError("head and wrist camera serial numbers must differ")
    if cfg.cameras.warmup_frames < 0:
        raise ValueError("cameras.warmup_frames must be non-negative")
    if cfg.robot.home_tcp_pose is not None and len(cfg.robot.home_tcp_pose) != 6:
        raise ValueError("robot.home_tcp_pose must contain six values")
    for name, value in (
        ("robot.rtde_frequency_hz", cfg.robot.rtde_frequency_hz),
        ("cameras.save_hz", cfg.cameras.save_hz),
        ("cameras.fps", cfg.cameras.fps),
        ("servoj.frequency_hz", cfg.servoj.frequency_hz),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
