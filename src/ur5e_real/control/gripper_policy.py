from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GripperCommandConfig:
    close_threshold: float = 0.60
    open_threshold: float = 0.40
    stable_count: int = 3
    minimum_command_interval_s: float = 2.0
    maximum_cycles: int | None = 1


class GripperPolicy:
    """Turn continuous 0=open/1=closed predictions into sparse serial commands."""

    def __init__(self, gripper: Any, config: GripperCommandConfig, *, initial_state: int = 0) -> None:
        if initial_state not in (0, 1):
            raise ValueError("initial commanded gripper state must be 0 or 1")
        self.gripper = gripper
        self.config = config
        self.state = "wait_open" if initial_state else "wait_close"
        self.close_streak = 0
        self.open_streak = 0
        self.last_command_at = 0.0
        self.cycles = 0
        self.estimated = float(initial_state)

    def step(self, predicted: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.config.maximum_cycles is not None and self.cycles >= self.config.maximum_cycles:
            return
        interval_ok = now - self.last_command_at >= self.config.minimum_command_interval_s
        if self.state == "wait_close":
            self.close_streak = self.close_streak + 1 if predicted >= self.config.close_threshold else 0
            if self.close_streak >= self.config.stable_count and interval_ok:
                self.gripper.close()
                self.last_command_at = now
                self.estimated = 1.0
                self.state = "wait_open"
                self.open_streak = 0
        elif self.state == "wait_open":
            self.open_streak = self.open_streak + 1 if predicted <= self.config.open_threshold else 0
            if self.open_streak >= self.config.stable_count and interval_ok:
                self.gripper.open()
                self.last_command_at = now
                self.estimated = 0.0
                self.cycles += 1
                done = self.config.maximum_cycles is not None and self.cycles >= self.config.maximum_cycles
                self.state = "done" if done else "wait_close"
                self.close_streak = 0
