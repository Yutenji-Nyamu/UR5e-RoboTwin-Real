from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Any


class TerminalKeyPoller:
    def __init__(self) -> None:
        self.enabled = False
        self._fd: int | None = None
        self._old: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyPoller":
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        attributes = termios.tcgetattr(self._fd)
        attributes[3] &= ~termios.ECHO
        termios.tcsetattr(self._fd, termios.TCSADRAIN, attributes)
        self.enabled = True
        return self

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        value = sys.stdin.read(1)
        return value.lower() if value else None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.enabled and self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
