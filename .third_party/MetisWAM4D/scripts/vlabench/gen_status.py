"""Progress of ``launch_generate.sh`` from the per-worker jsonl logs, and the training index once complete.

    python scripts/vlabench/gen_status.py [--out ROOT] [--index]

``--index`` writes ``<out>/index.jsonl`` (task, episode, frames, split) over the finished episode files; the last
``VAL_PER_TASK`` episode indices of each task (``episode >= EPISODES - VAL_PER_TASK``) are the held-out split.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

EPISODES = 500
VAL_PER_TASK = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/ytech_milm_intern/danglingwei/datas/VLABench/gen4d")
    ap.add_argument("--index", action="store_true")
    args = ap.parse_args()
    root = Path(args.out)
    stats = collections.defaultdict(lambda: collections.Counter())
    land, frames = collections.defaultdict(list), {}
    for log in sorted((root / "_logs").glob("*.jsonl")):
        for line in open(log):
            r = json.loads(line)
            s = stats[r["task"]]
            s["attempts"] += 1
            s["ok" if r["success"] else ("error" if "error" in r else "expert_fail")] += 1
            if r["success"]:
                land[r["task"]].append(r.get("land_within_5mm", float("nan")))
                frames[(r["task"], r["episode"])] = r["frames"]
                s["sec"] += r.get("t_total", 0)
    total = 0
    print(f"{'task':24s} {'ok':>5s} {'att':>5s} {'rate':>5s} {'err':>4s} {'s/ep':>5s} {'<5mm':>5s}")
    for task in sorted(stats):
        s = stats[task]
        total += s["ok"]
        lw = sum(land[task]) / max(len(land[task]), 1)
        print(f"{task:24s} {s['ok']:5d} {s['attempts']:5d} {s['ok'] / max(s['attempts'], 1):5.2f} {s['error']:4d} "
              f"{s['sec'] / max(s['ok'], 1):5.0f} {lw:5.3f}")
    print(f"total ok {total}")
    if args.index:
        import h5py
        rows = []
        for path in sorted(root.glob("*/episode*.h5")):
            task, ep = path.parent.name, int(path.stem[len("episode"):])
            with h5py.File(path) as h:   # a helper and the original worker may both have written episode k
                n = int(h.attrs["frames"])
            rows.append(dict(task=task, episode=ep, frames=n,
                             split="val" if ep >= EPISODES - VAL_PER_TASK else "train"))
        with open(root / "index.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"index: {len(rows)} episodes, val {sum(r['split'] == 'val' for r in rows)}")


if __name__ == "__main__":
    main()
