"""Opt-in RealSense RGB-D capture; independent of the existing RGB interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from .realsense import _rs_module


@dataclass(frozen=True)
class RgbdPair:
    head: Any
    wrist: Any
    head_depth: Any
    wrist_depth: Any
    metadata: dict


def _intrinsics(profile):
    value = profile.as_video_stream_profile().get_intrinsics()
    return {
        key: getattr(value, key) for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs")
    } | {"distortion_model": str(value.model)}


def _stamp(frame):
    return {
        "frame_number": frame.get_frame_number(),
        "timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
    }


class DualRgbdCamera:
    def __init__(self, head_serial, wrist_serial, width, height, fps, warmup_frames=60):
        self.serials = {"head": head_serial, "wrist": wrist_serial}
        self.width, self.height, self.fps = width, height, fps
        self.warmup_frames = warmup_frames
        self.pipelines: dict[str, Any] = {}
        self.aligners: dict[str, Any] = {}
        self.calibration: dict = {}
        self.previous: dict = {}

    def start(self):
        if self.pipelines:
            raise RuntimeError("RGB-D cameras already started")
        rs = _rs_module()
        self.previous.clear()
        self.calibration = {
            "version": 1,
            "alignment": "depth_to_color",
            "encoding": "z16_uint16",
            "invalid_depth_value": 0,
            "extrinsics_rotation_layout": "column_major",
            "cameras": {},
        }
        try:
            for role, serial in self.serials.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                profile = pipeline.start(config)
                self.pipelines[role] = pipeline
                self.aligners[role] = rs.align(rs.stream.color)
                device = profile.get_device()
                scale = device.first_depth_sensor().get_depth_scale()
                if not math.isfinite(scale) or scale <= 0:
                    raise RuntimeError(f"{role}: invalid depth scale")
                color, depth = profile.get_stream(rs.stream.color), profile.get_stream(rs.stream.depth)
                extrinsic = depth.get_extrinsics_to(color)
                camera = {
                    "serial": serial,
                    "depth_scale_m_per_unit": scale,
                    "color_intrinsics": _intrinsics(color),
                    "native_depth_intrinsics": _intrinsics(depth),
                    "depth_to_color": {
                        "rotation": list(extrinsic.rotation), "translation_m": list(extrinsic.translation),
                    },
                    "streams": {
                        "color": {"width": self.width, "height": self.height, "fps": color.fps(),
                                  "format": str(color.format())},
                        "depth": {"width": self.width, "height": self.height, "fps": depth.fps(),
                                  "format": str(depth.format())},
                    },
                }
                for key in ("name", "firmware_version", "usb_type_descriptor"):
                    info = getattr(rs.camera_info, key)
                    camera[key] = device.get_info(info) if device.supports(info) else None
                self.calibration["cameras"][role] = camera
            for _ in range(self.warmup_frames):
                for pipeline in self.pipelines.values():
                    pipeline.wait_for_frames(2000)
            # Validate actual RGB + depth and aligned calibration before the collector
            # is allowed to start freedrive. Discard this startup observation.
            self.read()
        except BaseException:
            self.stop()
            raise

    def read(self) -> RgbdPair:
        import numpy as np

        if set(self.pipelines) != {"head", "wrist"}:
            raise RuntimeError("RGB-D cameras are not started")
        images, depths, metadata = {}, {}, {}
        for role, pipeline in self.pipelines.items():
            frames = pipeline.wait_for_frames(2000)
            receive_wall, receive_mono = time.time(), time.monotonic()
            native_depth = frames.get_depth_frame()
            native_color = frames.get_color_frame()
            if not native_depth or not native_color:
                raise RuntimeError(f"{role}: missing RGB-D frame; refusing RGB-only fallback")
            stamps = {"color": _stamp(native_color), "depth": _stamp(native_depth)}
            for stream, stamp in stamps.items():
                if not math.isfinite(stamp["timestamp_ms"]):
                    raise RuntimeError(f"{role}/{stream}: invalid camera timestamp")
                previous = self.previous.get((role, stream))
                if previous is not None and (
                    stamp["frame_number"] <= previous["frame_number"]
                    or stamp["timestamp_ms"] <= previous["timestamp_ms"]
                    or stamp["timestamp_domain"] != previous["timestamp_domain"]
                ):
                    raise RuntimeError(f"{role}/{stream}: repeated frame or nonmonotonic camera clock")
            aligned = self.aligners[role].process(frames)
            depth, color = aligned.get_depth_frame(), aligned.get_color_frame()
            if not depth or not color:
                raise RuntimeError(f"{role}: incomplete aligned RGB-D frame")
            images[role] = np.asanyarray(color.get_data()).copy()
            depths[role] = np.asanyarray(depth.get_data()).copy()
            if images[role].dtype != np.uint8 or images[role].shape != (self.height, self.width, 3):
                raise RuntimeError(f"{role}: invalid color image")
            if depths[role].dtype != np.uint16 or depths[role].shape != (self.height, self.width):
                raise RuntimeError(f"{role}: invalid Z16 depth image")
            self.calibration["cameras"][role]["aligned_depth_intrinsics"] = _intrinsics(depth.profile)
            metadata[role] = {
                **stamps,
                "host_receive_time_s": receive_wall,
                "host_receive_monotonic_s": receive_mono,
                "valid_depth_fraction": float(np.count_nonzero(depths[role]) / depths[role].size),
            }
            for stream, stamp in stamps.items():
                self.previous[role, stream] = stamp
        return RgbdPair(images["head"], images["wrist"], depths["head"], depths["wrist"], metadata)

    def stop(self):
        for pipeline in self.pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        self.pipelines.clear()
        self.aligners.clear()
