"""Enumerate RealSense devices and optionally verify one synchronized frame pair."""

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.realsense import DualColorCamera, list_serials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    for serial in list_serials():
        print(serial)
    if args.frames <= 0:
        return
    if not args.config:
        parser.error("--config is required when --frames is greater than zero")
    cfg = load_config(args.config)
    cameras = DualColorCamera(
        cfg.cameras.head_serial,
        cfg.cameras.wrist_serial,
        cfg.cameras.width,
        cfg.cameras.height,
        cfg.cameras.fps,
    )
    try:
        cameras.start()
        for index in range(args.frames):
            pair = cameras.read()
            if pair is None:
                raise RuntimeError("camera pair returned no color frame")
            print(f"frame {index + 1}: head={pair.head.shape}; wrist={pair.wrist.shape}")
    finally:
        cameras.stop()


if __name__ == "__main__":
    main()
