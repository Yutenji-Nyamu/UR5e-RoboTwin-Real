"""Read PolyScope/Dashboard status without changing robot state."""

from __future__ import annotations

import argparse
import socket

from ur5e_real.config import load_config


QUERIES = (
    "PolyscopeVersion",
    "robotmode",
    "safetystatus",
    "programState",
    "is in remote control",
    "get operational mode",
)


def query_status(host: str, port: int = 29999, timeout_s: float = 3.0) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    with socket.create_connection((host, port), timeout=timeout_s) as connection:
        connection.settimeout(timeout_s)
        stream = connection.makefile("rw", encoding="utf-8", newline="\n")
        results.append(("connected", stream.readline().strip()))
        for query in QUERIES:
            stream.write(f"{query}\n")
            stream.flush()
            results.append((query, stream.readline().strip()))
        stream.write("quit\n")
        stream.flush()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name, value in query_status(cfg.robot.host):
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
