from __future__ import annotations

import importlib.util
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import LabConfig


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _module(name: str) -> Check:
    found = importlib.util.find_spec(name) is not None
    return Check(f"python:{name}", found, "available" if found else "missing")


def _tcp(host: str, port: int, timeout: float = 0.5) -> Check:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return Check(f"tcp:{host}:{port}", True, "reachable")
    except OSError as exc:
        return Check(f"tcp:{host}:{port}", False, str(exc))


def run_doctor(cfg: LabConfig, hardware: bool = False) -> list[Check]:
    checks = [
        Check("python", sys.version_info[:2] == (3, 10), sys.version.split()[0]),
        _module("yaml"),
        _module("numpy"),
        _module("cv2"),
        _module("h5py"),
        _module("serial"),
        _module("pyrealsense2"),
        _module("rtde.rtde"),
        Check("servoj:config_xml", cfg.servoj.config_xml.is_file(), str(cfg.servoj.config_xml)),
    ]
    if not hardware:
        return checks

    checks.extend(
        [
            _tcp(cfg.robot.host, cfg.robot.script_port),
            _tcp(cfg.robot.host, cfg.robot.rtde_port),
            Check("gripper:serial", Path(cfg.gripper.port).exists(), cfg.gripper.port),
        ]
    )
    try:
        from .hardware.realsense import list_serials

        serials = list_serials()
        checks.append(Check("camera:head", cfg.cameras.head_serial in serials, cfg.cameras.head_serial))
        checks.append(Check("camera:wrist", cfg.cameras.wrist_serial in serials, cfg.cameras.wrist_serial))
    except Exception as exc:  # hardware and vendor exceptions vary
        checks.append(Check("camera:enumeration", False, repr(exc)))
    return checks


def print_checks(checks: list[Check]) -> bool:
    for check in checks:
        mark = "OK" if check.ok else "FAIL"
        print(f"[{mark:4}] {check.name}: {check.detail}")
    return all(check.ok for check in checks)
