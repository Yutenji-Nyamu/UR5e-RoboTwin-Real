from __future__ import annotations

import socket
from collections.abc import Sequence


def send_urscript(program: str, host: str, port: int = 30001, timeout_s: float = 10.0) -> None:
    payload = program if program.endswith("\n") else f"{program}\n"
    with socket.create_connection((host, port), timeout=timeout_s) as connection:
        connection.sendall(payload.encode("utf-8"))


def start_freedrive(host: str, port: int = 30001, timeout_s: float = 10.0) -> None:
    send_urscript(
        """def ur5e_real_freedrive():
  freedrive_mode()
  while (True):
    sync()
  end
end
ur5e_real_freedrive()
""",
        host,
        port,
        timeout_s,
    )


def stop_freedrive(host: str, port: int = 30001, timeout_s: float = 10.0) -> None:
    send_urscript(
        """def ur5e_real_stop_freedrive():
  end_freedrive_mode()
end
ur5e_real_stop_freedrive()
""",
        host,
        port,
        timeout_s,
    )


def pose_literal(pose: Sequence[float]) -> str:
    if len(pose) != 6:
        raise ValueError("TCP pose must contain six values")
    return "p[" + ", ".join(f"{float(value):.9f}" for value in pose) + "]"


def speedl_command(
    velocity: Sequence[float], *, acceleration: float = 0.5, duration_s: float = 0.1
) -> str:
    if len(velocity) != 6:
        raise ValueError("TCP velocity must contain six values")
    if acceleration <= 0 or duration_s <= 0:
        raise ValueError("acceleration and duration must be positive")
    values = ", ".join(f"{float(value):.6f}" for value in velocity)
    return f"speedl([{values}], a={acceleration:.6f}, t={duration_s:.6f})\n"


def stopl_command(deceleration: float = 0.5) -> str:
    if deceleration <= 0:
        raise ValueError("deceleration must be positive")
    return f"stopl({deceleration:.6f})\n"


def move_linear(
    host: str,
    pose: Sequence[float],
    *,
    acceleration: float = 0.4,
    velocity: float = 0.04,
    blend_radius: float = 0.0,
    port: int = 30001,
    timeout_s: float = 10.0,
) -> None:
    if acceleration <= 0 or velocity <= 0 or blend_radius < 0:
        raise ValueError("acceleration and velocity must be positive; blend_radius must be non-negative")
    command = (
        f"movel({pose_literal(pose)}, a={acceleration:.6f}, "
        f"v={velocity:.6f}, r={blend_radius:.6f})"
    )
    send_urscript(command, host, port, timeout_s)
