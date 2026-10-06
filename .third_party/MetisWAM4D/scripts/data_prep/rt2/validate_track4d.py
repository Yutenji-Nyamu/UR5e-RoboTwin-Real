"""Validate RT2_MetisWAM4D episodes: open source.hdf5 / track4d.h5, check shapes against complete.json, and
(optionally) fully read the compressed Track4D datasets so corrupt chunks (lost page cache after a node
reboot) surface as errors.  Bad episodes are written to pending.jsonl-compatible rows.

  python validate_track4d.py --index index.jsonl --workers 32 [--full] [--tasks ...] [--rows rows.jsonl]
  python validate_track4d.py --rows candidates.jsonl --full --workers 16
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import h5py
import numpy as np

OUT = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D")


def check(row: dict, full: bool) -> str | None:
    d = OUT / row["task"] / row["variant"] / f"episode{row['episode']}"
    try:
        info = json.loads((d / "complete.json").read_text())
        frames = int(info["track4d"]["frames"])
        with h5py.File(d / "track4d.h5", "r") as f:
            for name, n in (("delta_xyz_cam", frames - 4), ("valid", frames - 4), ("role", frames)):
                if f[name].shape[0] != n:
                    return f"{name} has {f[name].shape[0]} frames, expected {n}"
            if full:
                for name in ("delta_xyz_cam", "valid", "role"):
                    ds = f[name]
                    for t in range(0, ds.shape[0], 16):
                        np.asarray(ds[t:t + 16])
            else:  # read the last chunk of every dataset (the part most likely lost at a reboot)
                for name in ("delta_xyz_cam", "valid", "role"):
                    np.asarray(f[name][-1])
        with h5py.File(d / "source.hdf5", "r") as src:
            if src["observation/head_camera/rgb"].shape[0] < frames:
                return "source.hdf5 shorter than track4d"
        return None
    except Exception as exc:  # noqa: BLE001
        return repr(exc)[:200]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=None)
    p.add_argument("--rows", default=None, help="jsonl rows to check instead of the index")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--full", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=str(OUT / "_logs" / "validate_bad.jsonl"))
    args = p.parse_args()
    src = args.rows or (OUT / args.index)
    rows = [json.loads(l) for l in open(src) if l.strip()]
    if args.tasks:
        rows = [r for r in rows if r["task"] in args.tasks]
    bad = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(check, r, args.full): r for r in rows}
        for i, fut in enumerate(as_completed(futures)):
            err = fut.result()
            if err:
                r = futures[fut]
                bad.append({**{k: r[k] for k in ("task", "variant", "episode")}, "error": err})
                print(f"BAD {r['task']}/{r['variant']}/episode{r['episode']}: {err}", flush=True)
            if (i + 1) % 1000 == 0:
                print(f"checked {i + 1}/{len(rows)}, bad {len(bad)}", flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        for b in bad:
            f.write(json.dumps(b) + "\n")
    print(f"checked {len(rows)}: {len(bad)} bad -> {args.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
