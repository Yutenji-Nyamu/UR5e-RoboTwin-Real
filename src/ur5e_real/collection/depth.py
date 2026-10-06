"""Lossless depth files and camera timestamps alongside the unchanged RGB sync CSV."""

from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path


class DepthWriter:
    def __init__(self, camera_dir: Path, calibration: dict):
        self.directories = {role: camera_dir / f"{role}_depth" for role in ("head", "wrist")}
        for directory in self.directories.values():
            directory.mkdir(exist_ok=False)
        try:
            sdk_version = importlib.metadata.version("pyrealsense2")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        calibration = {
            **calibration,
            "sdk_version": sdk_version,
            "depth_to_meters": "uint16_value * depth_scale_m_per_unit",
            "time_relation": "SDK timestamps and host receipt; no robot/camera hardware sync claimed",
        }
        with (camera_dir / "camera_calibration.json").open("x", encoding="utf-8") as handle:
            json.dump(calibration, handle, indent=2)
            handle.write("\n")
        self.handle = (camera_dir / "rgbd_frames.csv").open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=[
            "frame_idx", "controller_time_s", "camera", "color_image", "depth_image",
            "color_frame_number", "depth_frame_number", "color_timestamp_ms", "depth_timestamp_ms",
            "color_timestamp_domain", "depth_timestamp_domain", "host_receive_time_s",
            "host_receive_monotonic_s", "valid_depth_fraction",
        ])
        try:
            self.writer.writeheader()
            self.handle.flush()
        except BaseException:
            self.handle.close()
            raise

    def write(self, frame_idx: int, controller_time_s: float, pair):
        import cv2
        import numpy as np

        name = f"frame_{frame_idx:05d}.png"
        rows = []
        # Publish no sidecar rows until BOTH images have been encoded successfully.
        for role, directory in self.directories.items():
            depth = getattr(pair, f"{role}_depth")
            if depth.dtype != np.uint16 or depth.ndim != 2 or depth.shape != getattr(pair, role).shape[:2]:
                raise ValueError(f"{role}: depth must be aligned single-channel uint16")
            path = directory / name
            if path.exists():
                raise FileExistsError(path)
            if not cv2.imwrite(str(path), depth):
                raise RuntimeError(f"failed to write {role} depth frame")
            meta = pair.metadata[role]
            rows.append({
                "frame_idx": frame_idx, "controller_time_s": controller_time_s, "camera": role,
                "color_image": f"{role}/{name}", "depth_image": f"{role}_depth/{name}",
                **{f"{stream}_{key}": meta[stream][key]
                   for stream in ("color", "depth") for key in ("frame_number", "timestamp_ms", "timestamp_domain")},
                **{key: meta[key] for key in ("host_receive_time_s", "host_receive_monotonic_s",
                                              "valid_depth_fraction")},
            })
        self.writer.writerows(rows)
        self.handle.flush()

    def close(self):
        self.handle.close()
