from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _rs_module() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for RealSense support") from exc
    return rs


def list_serials() -> list[str]:
    rs = _rs_module()
    context = rs.context()
    return [device.get_info(rs.camera_info.serial_number) for device in context.query_devices()]


def start_color_pipeline(serial: str, width: int, height: int, fps: int) -> Any:
    rs = _rs_module()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


@dataclass(frozen=True)
class FramePair:
    head: Any
    wrist: Any


class DualColorCamera:
    def __init__(self, head_serial: str, wrist_serial: str, width: int, height: int, fps: int) -> None:
        self.head_serial = head_serial
        self.wrist_serial = wrist_serial
        self.width = width
        self.height = height
        self.fps = fps
        self.head_pipeline: Any = None
        self.wrist_pipeline: Any = None

    def start(self) -> None:
        available = list_serials()
        missing = [serial for serial in (self.head_serial, self.wrist_serial) if serial not in available]
        if missing:
            raise RuntimeError(f"RealSense devices not found: {missing}; available={available}")
        self.head_pipeline = start_color_pipeline(self.head_serial, self.width, self.height, self.fps)
        try:
            self.wrist_pipeline = start_color_pipeline(self.wrist_serial, self.width, self.height, self.fps)
        except Exception:
            self.head_pipeline.stop()
            self.head_pipeline = None
            raise

    def read(self) -> FramePair | None:
        if self.head_pipeline is None or self.wrist_pipeline is None:
            raise RuntimeError("cameras are not started")
        head_frame = self.head_pipeline.wait_for_frames().get_color_frame()
        wrist_frame = self.wrist_pipeline.wait_for_frames().get_color_frame()
        if not head_frame or not wrist_frame:
            return None
        import numpy as np

        return FramePair(np.asanyarray(head_frame.get_data()), np.asanyarray(wrist_frame.get_data()))

    def stop(self) -> None:
        for pipeline in (self.head_pipeline, self.wrist_pipeline):
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
        self.head_pipeline = None
        self.wrist_pipeline = None

    def __enter__(self) -> "DualColorCamera":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
