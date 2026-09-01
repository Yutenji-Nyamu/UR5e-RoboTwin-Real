"""Minimal URScript moveL check. Does nothing unless --execute is supplied."""

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.urscript import move_linear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pose", type=float, nargs=6, required=True, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--acceleration", type=float, default=0.4)
    parser.add_argument("--velocity", type=float, default=0.04)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"target pose: {args.pose}; execute={args.execute}")
    if args.execute:
        move_linear(
            cfg.robot.host,
            args.pose,
            acceleration=args.acceleration,
            velocity=args.velocity,
            port=cfg.robot.script_port,
            timeout_s=cfg.robot.socket_timeout_s,
        )


if __name__ == "__main__":
    main()
