#!/usr/bin/env python3
"""EEF20 normalisation statistics of the IG-10K robot subset.

Reads every LeRobot data file of the preprocessed robot directories, converts the 14-D Realman
action rows to Alpha's EEF20 (``metiswam4d.data.human.ig10k_action``) and writes per-dimension
min / max / q01 / q99 / mean / std to ``<root>/action_stats_eef20.json``.

    python3 scripts/data_prep/ig10k_robot_action_stats.py [--root ROOT] [--out PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metiswam4d.data.human.ig10k_action import ACTION_COLUMN, realman_to_eef20  # noqa: E402

ROOT = "/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/robot"
NAMES = [f"{arm}_{n}" for arm in ("left", "right")
         for n in ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    files = sorted(glob.glob(str(root / "imitator_robot_v1" / "*" / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise SystemExit(f"no data files under {root}")
    chunks, per_dir = [], {}
    for f in files:
        raw = np.stack(pq.read_table(f, columns=[ACTION_COLUMN]).column(ACTION_COLUMN).to_numpy(zero_copy_only=False))
        eef = realman_to_eef20(raw.astype(np.float32))
        chunks.append(eef)
        per_dir[Path(f).parents[2].name] = int(len(eef))
    eef = np.concatenate(chunks)
    stats = {
        "source": ACTION_COLUMN, "layout": "alpha_eef20", "names": NAMES, "rows": int(len(eef)),
        "directories": len(per_dir),
        "min": eef.min(0).tolist(), "max": eef.max(0).tolist(),
        "q01": np.percentile(eef, 1, axis=0).tolist(), "q99": np.percentile(eef, 99, axis=0).tolist(),
        "mean": eef.mean(0).tolist(), "std": eef.std(0).tolist(),
    }
    out = Path(args.out) if args.out else root / "action_stats_eef20.json"
    out.write_text(json.dumps(stats, indent=1))
    print(f"{len(eef)} rows from {len(per_dir)} directories -> {out}")
    for name, lo, hi, q1, q99 in zip(NAMES, stats["min"], stats["max"], stats["q01"], stats["q99"]):
        print(f"  {name:14s} min {lo:8.3f}  q01 {q1:8.3f}  q99 {q99:8.3f}  max {hi:8.3f}")


if __name__ == "__main__":
    main()
