"""Training root over several gen4d generation outputs: tasks present only in the base root are linked as directories,
tasks with extra episodes get a directory of per-episode links; ``index.jsonl`` keeps the base rows (train / val) and
adds the extra episodes as train rows; ``--repeat task=k`` writes every train row of a task k times (sampling weight).

    /usr/bin/python3.10 scripts/vlabench/build_mix_root.py --base <gen4d_v2> --extra <gen4d_v3> --out <root> \
        --repeat insert_flower=2,select_painting=2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py

SHARED = ("text_cache", "depth_stats.json", "uvd_stats.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", default="")
    args = ap.parse_args()
    base, extra, out = Path(args.base), Path(args.extra), Path(args.out)
    repeat = {k: int(v) for k, v in (kv.split("=") for kv in args.repeat.split(",") if kv)}
    out.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        if not (out / name).is_symlink():
            os.symlink((base / name).resolve(), out / name)

    rows = [json.loads(line) for line in (base / "index.jsonl").read_text().splitlines() if line.strip()]
    tasks = sorted({r["task"] for r in rows})
    extra_rows = []
    for task in tasks:
        files = sorted((extra / task).glob("episode*.h5")) if (extra / task).is_dir() else []
        target = out / task
        if not files:
            if not target.is_symlink():
                os.symlink((base / task).resolve(), target)
            continue
        target.mkdir(exist_ok=True)
        for src in [base / task / f"episode{r['episode']}.h5" for r in rows if r["task"] == task] + files:
            link = target / src.name
            if not link.is_symlink():
                os.symlink(src.resolve(), link)
        for f in files:
            with h5py.File(f) as h:
                extra_rows.append({"task": task, "episode": int(f.stem[7:]), "frames": int(h.attrs["frames"]),
                                   "split": "train"})
    counts = {}
    with open(out / "index.jsonl.tmp", "w") as handle:
        for r in rows + extra_rows:
            n = repeat.get(r["task"], 1) if r["split"] == "train" else 1
            for _ in range(n):
                handle.write(json.dumps(r) + "\n")
            counts.setdefault(r["task"], [0, 0])[0 if r["split"] == "train" else 1] += n
    os.replace(out / "index.jsonl.tmp", out / "index.jsonl")
    summary = {"base": str(base), "extra": str(extra), "repeat": repeat, "extra_episodes": len(extra_rows),
               "rows_train_val": counts}
    (out / "mix_root.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
