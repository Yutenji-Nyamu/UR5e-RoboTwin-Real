from __future__ import annotations

import time
from typing import Any

MOTOR_OPEN_CMD = bytes([0x02, 0x00, 0x20, 0x2F, 0x00, 0x00, 0xA4])
MOTOR_CLOSE_CMD = bytes([0x02, 0x01, 0x20, 0x2F, 0x00, 0x00, 0xA4])


class GripperSerial:
    def __init__(self, port: str, baudrate: int = 9600, timeout_s: float = 1.0) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for gripper control") from exc
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.serial: Any = serial.Serial(port, baudrate, timeout=timeout_s)
        time.sleep(0.2)

    def _write(self, payload: bytes) -> None:
        self.serial.write(payload)
        self.serial.flush()

    def open(self) -> None:
        self._write(MOTOR_OPEN_CMD)

    def close(self) -> None:
        self._write(MOTOR_CLOSE_CMD)

    def shutdown(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None

    def __enter__(self) -> "GripperSerial":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown()
