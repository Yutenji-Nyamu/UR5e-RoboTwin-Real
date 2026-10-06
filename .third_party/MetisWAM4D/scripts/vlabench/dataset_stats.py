"""Head-depth range for the depth condition (``depth_stats.json``) and Track uvd statistics (``uvd_stats.json``) over a
random subset of generated VLABench episodes.

    /usr/bin/python3.10 scripts/vlabench/dataset_stats.py [--root ROOT] [--episodes 200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metiswam4d.data.rt2.codec import UVD_SCALE  # noqa: E402

WORKSPACE_MAX_M = 2.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ytech_milm_intern/danglingwei/datas/VLABench/gen4d")
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()
    root = Path(args.root)
    paths = sorted(root.glob("*/episode*.h5"))
    random.Random(0).shuffle(paths)
    paths = paths[:args.episodes]
    depth, uvd = [], {1: [], 2: []}
    for path in paths:
        with h5py.File(path) as h:
            n = h["delta_uvd"].shape[0]
            for t in np.linspace(0, n - 1, 4).astype(int):
                d = h["head_depth_mm"][t].astype(np.float32) / 1000.0
                depth.append(d[d > 0][::7])
                role, delta = h["role"][t], h["delta_uvd"][t].astype(np.float32)
                for r in (1, 2):
                    uvd[r].append(np.abs(delta[role == r]))
    depth = np.concatenate(depth)
    q = {f"p{p}": float(np.percentile(depth, p)) for p in (0, 0.1, 1, 50, 90, 95, 99, 99.9, 100)}
    (root / "depth_stats.json").write_text(json.dumps({
        "head_camera": {"min_m": q["p0.1"], "max_m": WORKSPACE_MAX_M, "p50_m": q["p50"]},
        "scope": (f"full-scene z-depth of the forward camera; {len(paths)} random episodes x 4 frames. The depth is "
                  f"bimodal (table / robot ~1-1.5 m, back wall ~4.2 m, > 10 % of pixels): max = {WORKSPACE_MAX_M} m "
                  "so the wall saturates and the workspace keeps the 8-bit resolution"),
        "quantiles_m": q, "episodes": len(paths)}, indent=1) + "\n")
    width = 320
    stats = {}
    for r, name in ((1, "robot"), (2, "object")):
        a = np.concatenate(uvd[r]) if uvd[r] else np.zeros((0, 3))
        scale = UVD_SCALE * np.array([width, width, 1.0])
        stats[name] = {
            "pixels": int(len(a)),
            "p50": np.percentile(a, 50, axis=0).round(4).tolist() if len(a) else None,
            "p99": np.percentile(a, 99, axis=0).round(4).tolist() if len(a) else None,
            "p99.9": np.percentile(a, 99.9, axis=0).round(4).tolist() if len(a) else None,
            "clip_rate": (a > scale).mean(axis=0).round(5).tolist() if len(a) else None,
        }
    (root / "uvd_stats.json").write_text(json.dumps({"units": "|du| px, |dv| px, |dd| m (240x320 grid)",
                                                     "codec_scale": [width / 6, width / 6, 0.10], **stats},
                                                    indent=1) + "\n")
    print(json.dumps(q), json.dumps(stats), sep="\n")


if __name__ == "__main__":
    main()
