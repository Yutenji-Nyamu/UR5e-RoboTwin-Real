"""Development track for inference choices (execute steps, checkpoint window): the scene configs (``env.save()`` at
generation) of the held-out gen4d episodes (index split ``val``, never trained on), written where the official client
loads a user-frozen track 5 (``VLABench/configs/evaluation/tracks/track_5_cross_task.json``).

    /usr/bin/python3.10 scripts/vlabench/build_dev_track.py --root <gen4d_v2>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

TRACK = Path("/ytech_milm_intern/danglingwei/files/VLABench/VLABench/configs/evaluation/tracks/track_5_cross_task.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines() if line.strip()]
    track: dict[str, list] = {}
    for r in sorted((r for r in rows if r["split"] == "val"), key=lambda r: (r["task"], r["episode"])):
        with h5py.File(root / r["task"] / f"episode{r['episode']}.h5") as h:
            track.setdefault(r["task"], []).append({"task": json.loads(h.attrs["episode_config"])["task"]})
    TRACK.write_text(json.dumps(track))
    print(json.dumps({t: len(v) for t, v in track.items()}), "->", TRACK)


if __name__ == "__main__":
    main()
