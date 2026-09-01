"""Minimal serial gripper check."""

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.gripper import GripperSerial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["open", "close"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"gripper {args.action}; execute={args.execute}")
    if not args.execute:
        return
    with GripperSerial(cfg.gripper.port, cfg.gripper.baudrate, cfg.gripper.timeout_s) as gripper:
        getattr(gripper, args.action)()


if __name__ == "__main__":
    main()
