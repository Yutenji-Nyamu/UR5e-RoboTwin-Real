"""Read a bounded number of UR5e TCP samples over RTDE."""

import argparse

from ur5e_real.config import load_config
from ur5e_real.hardware.rtde import RtdeOutputConfig, RtdeStateClient, RtdeTcpClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--joints", action="store_true", help="read q, qd, TCP and TCP offset from the same packet")
    args = parser.parse_args()
    cfg = load_config(args.config)
    client_type = RtdeStateClient if args.joints else RtdeTcpClient
    client = client_type(
        RtdeOutputConfig(cfg.robot.host, cfg.robot.rtde_port, cfg.robot.rtde_frequency_hz)
    )
    try:
        client.connect()
        for _ in range(args.samples):
            print(client.receive_state() if args.joints else client.receive())
    finally:
        client.close()


if __name__ == "__main__":
    main()
