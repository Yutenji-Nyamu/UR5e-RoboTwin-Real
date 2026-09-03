from __future__ import annotations

import socket

QUERIES = ("PolyscopeVersion", "robotmode", "safetystatus", "programState", "is in remote control")


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


def require_external_motion_ready(host: str) -> None:
    status = dict(query_status(host))
    failures: list[str] = []
    if not status.get("robotmode", "").upper().endswith("RUNNING"):
        failures.append(status.get("robotmode", "robot mode unavailable"))
    if not status.get("safetystatus", "").upper().endswith("NORMAL"):
        failures.append(status.get("safetystatus", "safety status unavailable"))
    if not status.get("programState", "").upper().startswith("STOPPED"):
        failures.append(status.get("programState", "program state unavailable"))
    if status.get("is in remote control", "").strip().lower() != "true":
        failures.append("PolyScope is not in Remote Control")
    if failures:
        raise RuntimeError("; ".join(failures))
