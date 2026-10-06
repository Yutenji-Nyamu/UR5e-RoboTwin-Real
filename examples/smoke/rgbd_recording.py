"""Test the production RGB-D camera/writer on disk, without connecting to a robot."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from ur5e_real.collection.depth import DepthWriter
from ur5e_real.config import load_config
from ur5e_real.hardware.rgbd import DualRgbdCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error("seconds must be in 1..600")
    cfg = load_config(args.config).cameras
    args.output.mkdir(parents=True, exist_ok=False)
    for role in ("head", "wrist"):
        (args.output / role).mkdir()
    camera = DualRgbdCamera(cfg.head_serial, cfg.wrist_serial, cfg.width, cfg.height, cfg.fps, cfg.warmup_frames)
    writer = None
    report = {"scope": "production camera + four PNG writes; no robot connection", "status": "initializing",
              "started_at": datetime.now(timezone.utc).isoformat(), "requested_seconds": args.seconds,
              "save_hz": cfg.save_hz, "pairs": 0, "depth_roundtrips": 0}
    times, durations, sizes = [], [], []
    try:
        camera.start()
        writer = DepthWriter(args.output, camera.calibration)
        start = deadline = time.monotonic()
        for idx in range(1, round(args.seconds * cfg.save_hz) + 1):
            time.sleep(max(0, deadline - time.monotonic()))
            begin = time.monotonic()
            pair = camera.read()
            for role in ("head", "wrist"):
                if not cv2.imwrite(str(args.output / role / f"frame_{idx:05d}.png"), getattr(pair, role)):
                    raise RuntimeError("RGB write failed")
            # No controller connection: NaN explicitly marks the absent robot clock.
            writer.write(idx, float("nan"), pair)
            times.append(begin - start)
            durations.append(time.monotonic() - begin)
            for role in ("head", "wrist"):
                path = args.output / f"{role}_depth" / f"frame_{idx:05d}.png"
                restored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if restored is None or restored.dtype != np.uint16 or not np.array_equal(restored, getattr(pair, f"{role}_depth")):
                    raise RuntimeError("depth file round trip failed")
                report["depth_roundtrips"] += 1
                sizes.append(path.stat().st_size)
            report["pairs"] = idx
            deadline += 1 / cfg.save_hz
        report.update({
            "status": "passed", "elapsed_s": time.monotonic() - start,
            "read_and_write_ms_p50_p95_max": [*np.percentile(durations, [50, 95]).tolist(), max(durations)],
            "frame_interval_ms_p50_p95_max": [*np.percentile(np.diff(times), [50, 95]).tolist(), max(np.diff(times))]
            if len(times) > 1 else None,
            "depth_png_bytes_mean": float(np.mean(sizes)),
            "png_bytes_total": sum(p.stat().st_size for p in args.output.glob("*/*.png")),
        })
        for key in ("read_and_write_ms_p50_p95_max", "frame_interval_ms_p50_p95_max"):
            if report[key] is not None:
                report[key] = [v * 1000 for v in report[key]]
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if writer:
            writer.close()
        camera.stop()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    with (args.output / "rgbd_frames.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == report["pairs"] * 2


if __name__ == "__main__":
    main()
