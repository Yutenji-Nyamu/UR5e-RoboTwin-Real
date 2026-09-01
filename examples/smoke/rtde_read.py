"""Read a bounded number of UR5e TCP samples over RTDE."""

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.rtde import RtdeOutputConfig, RtdeTcpClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)
    client = RtdeTcpClient(
        RtdeOutputConfig(cfg.robot.host, cfg.robot.rtde_port, cfg.robot.rtde_frequency_hz)
    )
    try:
        client.connect()
        for _ in range(args.samples):
            print(client.receive())
    finally:
        client.close()


if __name__ == "__main__":
    main()
