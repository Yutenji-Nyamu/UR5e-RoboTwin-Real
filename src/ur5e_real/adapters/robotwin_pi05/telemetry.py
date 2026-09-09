"""Bounded, best-effort resource sampling; never performs robot I/O."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_gpu_csv(text):
    keys = (
        "index",
        "uuid",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_pct",
        "power_w",
        "temperature_c",
    )
    rows = []
    for values in csv.reader(text.strip().splitlines()):
        if len(values) != len(keys):
            raise ValueError("unexpected nvidia-smi columns")
        item = {}
        for key, value in zip(keys, values):
            value = value.strip()
            try:
                item[key] = value if key == "uuid" else (float(value) if math.isfinite(float(value)) else None)
            except ValueError:
                item[key] = None
        rows.append(item)
    return rows


def sample_resources():
    result = {"gpus": [], "host": {}, "errors": []}
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True)
        result["gpus"] = parse_gpu_csv(completed.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["errors"].append(f"gpu: {type(exc).__name__}: {exc}")
    try:
        import psutil

        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        parent = psutil.Process(os.getpid())
        rss = parent.memory_info().rss
        child_rss = 0
        for child in parent.children(recursive=True):
            try:
                child_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        result["host"] = {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "used_bytes": vm.used,
            "process_rss_bytes": rss,
            "children_rss_sum_bytes": child_rss,
            "swap_used_bytes": swap.used,
            "swap_free_bytes": swap.free,
        }
    except (ImportError, OSError) as exc:
        result["errors"].append(f"host: {type(exc).__name__}: {exc}")
    return result


class ResourceMonitor:
    def __init__(self, directory: Path, *, interval_s=2.0, console_s=30.0, sampler=sample_resources):
        if not all(math.isfinite(value) and value > 0 for value in (interval_s, console_s)):
            raise ValueError("monitor intervals must be positive")
        self.directory = directory
        self.interval_s, self.console_s, self.sampler = interval_s, console_s, sampler
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase, self._step = "startup", 0
        self._started = time.monotonic()
        self._last_console = self._started
        self._thread = None
        self._file = None
        self.summary = {"samples": 0, "gpus": {}, "host": {}, "errors": [], "jax_peak_bytes": None}

    def phase(self, name, step=None):
        with self._lock:
            self._phase = name
            if step is not None:
                self._step = int(step)

    def jax_stats(self, devices):
        stats = []
        for device in devices:
            try:
                raw = device.memory_stats() or {}
                item = {
                    key: int(raw[key])
                    for key in (
                        "bytes_in_use",
                        "peak_bytes_in_use",
                        "bytes_limit",
                        "bytes_reserved",
                        "peak_bytes_reserved",
                    )
                    if raw.get(key) is not None
                }
                item["device"] = str(device)
                stats.append(item)
                peak = item.get("peak_bytes_in_use")
                if peak is not None:
                    with self._lock:
                        self.summary["jax_peak_bytes"] = max(self.summary["jax_peak_bytes"] or 0, peak)
            except (RuntimeError, AttributeError):
                stats.append({"device": str(device), "unavailable": True})
        return stats

    def sample(self):
        try:
            resource = self.sampler()
        except Exception as exc:  # A diagnostic failure must not kill training.
            resource = {"gpus": [], "host": {}, "errors": [f"sampler: {type(exc).__name__}: {exc}"]}
        with self._lock:
            item = {
                "time_utc": utc_now(),
                "elapsed_s": time.monotonic() - self._started,
                "phase": self._phase,
                "step": self._step,
                **resource,
            }
            self.summary["samples"] += 1
            for gpu in resource["gpus"]:
                peaks = self.summary["gpus"].setdefault(str(gpu.get("uuid")), {})
                for field, operation in (("memory_used_mib", max), ("memory_free_mib", min)):
                    value = gpu.get(field)
                    if value is not None:
                        peaks[field] = operation(peaks.get(field, value), value)
            for field, operation in (
                ("available_bytes", min),
                ("process_rss_bytes", max),
                ("children_rss_sum_bytes", max),
                ("swap_used_bytes", max),
            ):
                value = resource["host"].get(field)
                if value is not None:
                    self.summary["host"][field] = operation(self.summary["host"].get(field, value), value)
            for error in resource["errors"]:
                if error not in self.summary["errors"]:
                    self.summary["errors"].append(error)
                    print(f"[RESOURCE-WARNING] {error}", flush=True)
            if self._file is not None:
                self._file.write(json.dumps(item, allow_nan=False) + "\n")
            if time.monotonic() - self._last_console >= self.console_s:
                print("[RESOURCE] " + json.dumps(item, allow_nan=False), flush=True)
                self._last_console = time.monotonic()
        return item

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self.sample()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = (self.directory / "resources.jsonl").open("x", buffering=1)
        self.sample()
        self._thread = threading.Thread(target=self._run, name="pi05-resources", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
            if self._thread.is_alive():
                raise RuntimeError("resource sampler failed to terminate")
        self.sample()
        self._file.close()
        self.summary.update(
            {
                "sampled_peaks_only": True,
                "interval_s": self.interval_s,
                "elapsed_s": time.monotonic() - self._started,
                "status": "failed" if exc_type else "completed",
            }
        )
        with (self.directory / "resources_summary.json").open("x") as handle:
            json.dump(self.summary, handle, indent=2, allow_nan=False)
