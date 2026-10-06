"""Which tasks' scene description can be read without the expert's motion planning (``rt2_sim.scene_info_fast``).

Reference: scenes whose description the real expert produced (the seen instructions of the paired JanusAct4D-RT2
evaluation, reused as ``scene.json`` by the campaign).  Per task up to ``--per-task`` scenes are replayed with the
planner stubbed out; a task qualifies only if every description is identical and ``play_once`` returns in time.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /usr/bin/python3.10 scripts/eval/check_fast_scene_info.py --campaign <out> --log <jsonl>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import signal
import time


class Timeout(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    from metiswam4d.eval import rt2_sim
    rt2_sim.setup_runtime()
    jobs = json.loads((args.campaign / "campaign.json").read_text())["jobs"]
    picked = collections.defaultdict(list)
    for job in jobs:
        if job["episode"] >= 10 or len(picked[job["task"]]) >= args.per_task:
            continue
        path = args.campaign / "episodes" / job["key"] / "scene.json"
        if path.exists():
            scene = json.loads(path.read_text())
            if scene.get("source") and scene.get("info"):
                picked[job["task"]].append((job, scene))
    done = set()
    if args.log.exists():
        done = {json.loads(l)["key"] for l in args.log.read_text().splitlines() if l.strip()}

    def alarm(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, alarm)
    with open(args.log, "a") as log:
        for task in sorted(picked):
            for job, scene in picked[task]:
                if job["key"] in done:
                    continue
                started = time.time()
                row = {"key": job["key"], "task": task}
                signal.alarm(args.timeout)
                try:
                    info = rt2_sim.scene_info_fast(task, job["task_config"], int(scene["environment_seed"]),
                                                   int(job["protocol_episode_index"]))
                    row.update(match=info == scene["info"], fast=info, expert=scene["info"])
                except Timeout:
                    row.update(match=False, error="timeout")
                except Exception as exc:
                    row.update(match=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
                finally:
                    signal.alarm(0)
                row["seconds"] = round(time.time() - started, 1)
                log.write(json.dumps(row, default=str) + "\n")
                log.flush()
                print(json.dumps({k: row.get(k) for k in ("key", "match", "error", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
