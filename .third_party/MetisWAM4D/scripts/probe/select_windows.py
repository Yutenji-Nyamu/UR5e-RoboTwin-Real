#!/usr/bin/env python3
"""Pick the open-loop probe windows from the ``gt_events`` outputs.

For every processed episode we take (a) one window whose 32-frame future contains a key
interaction event (contact_start / motion_onset / liftoff / settle), with the event placed
in the middle half of the future so it is neither at the boundary nor already happened, and
(b) one window without any key event (approach / transport phase).  Windows are then
stratified per task/variant (``--per-variant`` of each kind) so that all 50 tasks weigh the
same.  Output: ``windows.jsonl`` with the dataset row index and window start, ready for
``ImperfectRoboTwinDataset.read_window``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

GT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes/gt_events")
BUCKETS = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1/buckets.tsv")
KEY = ("contact_start", "motion_onset", "liftoff", "settle")
WINDOW = 33


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--out", default=str(GT.parent / "windows.jsonl"))
    ap.add_argument("--per-variant", type=int, default=3, help="event windows AND no-event windows per task/variant")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = {(r["task"], r["variant"], r["episode"]): int(r["index"]) for r in csv.DictReader(open(BUCKETS), delimiter="\t")}
    by_variant: dict[tuple[str, str], dict[str, list]] = {}
    for f in sorted(Path(a.gt).glob("*/*/episode*.npz")):
        z = np.load(f, allow_pickle=False)
        task, variant, name = str(z["key"]).split("/")
        row = rows.get((task, variant, name[7:]))
        if row is None:
            continue
        T = int(z["frames"])
        events = [e for e in json.loads(str(z["events"])) if e["type"] in KEY]
        ev = np.array(sorted({e["frame"] for e in events}), dtype=int)
        bucket = by_variant.setdefault((task, variant), dict(event=[], quiet=[]))
        starts = list(range(1, T - WINDOW + 1))
        rng.shuffle(starts)
        got_e = got_q = 0
        for s in starts:
            inside = ev[(ev > s) & (ev <= s + 32)]
            mid = inside[(inside >= s + 8) & (inside <= s + 28)]
            rec = dict(row=row, start=s, key=f"{task}/{variant}/{name}", task=task, variant=variant,
                       events=[dict(type=e["type"], frame=e["frame"]) for e in events if s < e["frame"] <= s + 32])
            if len(mid) and got_e < 1:
                bucket["event"].append(rec)
                got_e += 1
            elif len(inside) == 0 and got_q < 1 and not ((ev > s - 8) & (ev <= s)).any():
                bucket["quiet"].append(rec)
                got_q += 1
            if got_e and got_q:
                break
    out = []
    for (task, variant), b in sorted(by_variant.items()):
        for kind in ("event", "quiet"):
            pool = b[kind]
            rng.shuffle(pool)
            for rec in pool[: a.per_variant]:
                out.append(dict(kind=kind, **rec))
    with open(a.out, "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    n_e = sum(r["kind"] == "event" for r in out)
    print(f"{len(out)} windows ({n_e} event / {len(out) - n_e} quiet) from {len(by_variant)} task-variants -> {a.out}")


if __name__ == "__main__":
    main()
