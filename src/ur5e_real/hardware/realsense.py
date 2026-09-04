from __future__ import annotations

import threading
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
    def __init__(
        self,
        head_serial: str,
        wrist_serial: str,
        width: int,
        height: int,
        fps: int,
        warmup_frames: int = 60,
    ) -> None:
        self.head_serial = head_serial
        self.wrist_serial = wrist_serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
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
            # D435i automatic exposure and white balance need initial frames to
            # converge. Never expose those startup frames to collection/inference.
            for _ in range(self.warmup_frames):
                self.head_pipeline.wait_for_frames()
                self.wrist_pipeline.wait_for_frames()
        except Exception:
            self.stop()
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


class LatestDualColorCamera:
    """Continuously acquire camera pairs so motion code never waits for a frame."""

    def __init__(self, camera: DualColorCamera) -> None:
        self.camera = camera
        self._latest: FramePair | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.camera.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ur5e-cameras", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.stop()
            raise RuntimeError("timed out waiting for the first camera pair")
        if self._error is not None:
            self.stop()
            raise RuntimeError("camera acquisition failed") from self._error

    def _loop(self) -> None:
        import numpy as np

        while self._running:
            try:
                pair = self.camera.read()
            except Exception as exc:
                if self._running:
                    self._error = exc
                    self._ready.set()
                return
            if pair is None:
                continue
            latest = FramePair(np.copy(pair.head), np.copy(pair.wrist))
            with self._lock:
                self._latest = latest
            self._ready.set()

    def read(self) -> FramePair | None:
        if self._error is not None:
            raise RuntimeError("camera acquisition failed") from self._error
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.camera.stop()
        self._thread = None
