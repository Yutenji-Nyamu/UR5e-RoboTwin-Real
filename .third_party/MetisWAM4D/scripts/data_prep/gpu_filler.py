"""Keep every visible GPU's utilisation high with real matmul work while bursty jobs share the device.

Unlike a sleep-kernel guard (a single tiny kernel whose measured utilisation collapses as soon as
another context time-slices the GPU), matmuls saturate the SMs for their slice; the device
utilisation stays near ``duty`` regardless of co-tenants.  One thread per GPU; ``--duty`` controls
the fraction of wall time spent computing (the rest is sleep so co-tenants get GPU time).

    /usr/bin/python3.10 scripts/data_prep/gpu_filler.py --duty 0.92 [--gpus 0,1,2] [--size 3072]
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import torch

STOP = threading.Event()


def worker(gpu: int, size: int, duty: float, queue: int) -> None:
    """Queue ``queue`` large matmuls per sync so the GPU stays busy even when this thread is
    CPU-starved (a saturated host must not translate into idle GPU time)."""
    device = torch.device("cuda", gpu)
    torch.cuda.set_device(device)
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(a)
    while not STOP.is_set():
        t0 = time.perf_counter()
        for _ in range(queue):
            torch.matmul(a, b, out=out)
        torch.cuda.synchronize(device)
        busy = time.perf_counter() - t0
        if duty < 1.0:
            time.sleep(busy * (1.0 - duty) / duty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--size", type=int, default=8192, help="matmul side; 8192^3 bf16 ~ 4 ms on A800")
    parser.add_argument("--duty", type=float, default=0.92)
    parser.add_argument("--queue", type=int, default=64, help="kernels queued per synchronisation")
    args = parser.parse_args()
    count = torch.cuda.device_count()
    gpus = list(range(count)) if args.gpus == "all" else [int(g) for g in args.gpus.split(",")]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    threads = [threading.Thread(target=worker, args=(g, args.size, args.duty, args.queue), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    print(f"gpu_filler on GPUs {gpus}: size={args.size} duty={args.duty}", flush=True)
    while not STOP.is_set():
        time.sleep(5)
    print("gpu_filler stopping", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
