"""Bounded dual RGB-D probe. Opens cameras only; never imports robot control.

Records calibration/timing/quality summaries and checks lossless Z16 PNG encoding.
Does not save scene images or change the production collector.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from ur5e_real.config import load_config


def intrinsics(profile):
    value = profile.as_video_stream_profile().get_intrinsics()
    return {
        key: getattr(value, key)
        for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs")
    } | {"distortion_model": str(value.model)}


def frame_metadata(frame):
    return {
        "frame_number": frame.get_frame_number(),
        "timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 300:
        parser.error("--frames must be in 1..300")
    if args.output.exists():
        parser.error("refusing to overwrite existing evidence")
    cfg = load_config(args.config).cameras
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "camera-only sequential dual read; no RTDE, gripper or freedrive",
        "versions": {p: importlib.metadata.version(p) for p in ("pyrealsense2", "numpy", "opencv-python")},
        "requested_profile": {"width": cfg.width, "height": cfg.height, "fps": cfg.fps},
        "warmup_pairs": cfg.warmup_frames,
        "requested_pairs": args.frames,
        "cameras": {},
        "samples": [],
        "status": "initializing",
    }
    pipelines = {}
    aligners = {}
    pair_ms = []
    try:
        for role, serial in (("head", cfg.head_serial), ("wrist", cfg.wrist_serial)):
            pipe = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            profile = pipe.start(config)
            pipelines[role] = pipe
            aligners[role] = rs.align(rs.stream.color)
            device = profile.get_device()
            color = profile.get_stream(rs.stream.color)
            depth = profile.get_stream(rs.stream.depth)
            extr = depth.get_extrinsics_to(color)
            report["cameras"][role] = {
                "serial": serial,
                "name": device.get_info(rs.camera_info.name),
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "usb": device.get_info(rs.camera_info.usb_type_descriptor),
                "depth_scale_m_per_unit": device.first_depth_sensor().get_depth_scale(),
                "color_intrinsics": intrinsics(color),
                "native_depth_intrinsics": intrinsics(depth),
                "depth_to_color": {"rotation": extr.rotation, "translation_m": extr.translation},
            }
        for _ in range(cfg.warmup_frames):
            for pipe in pipelines.values():
                pipe.wait_for_frames(5000)
        for index in range(args.frames):
            start = time.monotonic()
            sample = {"index": index}
            for role, pipe in pipelines.items():
                frames = pipe.wait_for_frames(5000)
                host_received = time.time()
                raw_depth = frames.get_depth_frame()
                begin_align = time.monotonic()
                aligned = aligners[role].process(frames)
                align_ms = (time.monotonic() - begin_align) * 1000
                depth, color = aligned.get_depth_frame(), aligned.get_color_frame()
                if not raw_depth or not depth or not color:
                    raise RuntimeError(f"{role}: incomplete RGB-D frameset")
                array = np.asanyarray(depth.get_data())
                if array.dtype != np.uint16 or array.shape != (cfg.height, cfg.width):
                    raise RuntimeError(f"{role}: invalid aligned depth shape/dtype")
                valid = array[array > 0]
                scale = report["cameras"][role]["depth_scale_m_per_unit"]
                item = {
                    "host_receive_time_s": host_received,
                    "color": frame_metadata(color),
                    "native_depth": frame_metadata(raw_depth),
                    "aligned_depth": frame_metadata(depth),
                    "align_ms": align_ms,
                    "valid_fraction": float(np.mean(array > 0)),
                    "valid_depth_m_p05_p50_p95": (np.percentile(valid, [5, 50, 95]) * scale).tolist()
                    if valid.size else None,
                }
                if index == 0:
                    begin_encode = time.monotonic()
                    ok, png = cv2.imencode(".png", array)
                    encode_ms = (time.monotonic() - begin_encode) * 1000
                    restored = cv2.imdecode(png, cv2.IMREAD_UNCHANGED) if ok else None
                    if restored is None or restored.dtype != np.uint16 or not np.array_equal(array, restored):
                        raise RuntimeError(f"{role}: lossless depth PNG round trip failed")
                    report["cameras"][role].update({
                        "aligned_depth_intrinsics": intrinsics(depth.profile),
                        "color_shape": list(np.asanyarray(color.get_data()).shape),
                        "depth_shape": list(array.shape),
                        "depth_dtype": str(array.dtype),
                        "first_depth_png_bytes": int(png.size),
                        "first_depth_png_encode_ms": encode_ms,
                        "png_roundtrip_exact": True,
                    })
                sample[role] = item
            report["samples"].append(sample)
            pair_ms.append((time.monotonic() - start) * 1000)
        report["pair_loop_ms_p50_p95_max"] = [*np.percentile(pair_ms, [50, 95]).tolist(), max(pair_ms)]
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for pipe in pipelines.values():
            try:
                pipe.stop()
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(str(exc))
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
