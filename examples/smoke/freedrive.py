"""Minimal freedrive start/stop check."""

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.urscript import start_freedrive, stop_freedrive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"freedrive {args.action}; execute={args.execute}")
    if not args.execute:
        return
    operation = start_freedrive if args.action == "start" else stop_freedrive
    operation(cfg.robot.host, cfg.robot.script_port, cfg.robot.socket_timeout_s)


if __name__ == "__main__":
    main()
