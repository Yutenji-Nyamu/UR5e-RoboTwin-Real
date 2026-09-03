"""Read PolyScope/Dashboard status without changing robot state."""

from __future__ import annotations

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.dashboard import query_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name, value in query_status(cfg.robot.host):
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
