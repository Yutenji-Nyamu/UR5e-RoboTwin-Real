"""420-episode protocol table of a multi-model fusion: every task config takes its episodes from one campaign.

    PYTHONPATH=. /usr/bin/python3.10 metiswam4d_inspired_by_internw0/scripts/fusion_report.py \
        --reference <old 420-episode campaign> \
        --source <campaign dir>[:task1,task2,...] [--source ...] --out <report dir>

Sources are applied in order (later ones override earlier ones for their task configs; without a task list a source
covers every config it contains).  Only protocol episodes count: the first N in layout order, N = the reference episode
count of the config (standalone 10, Generalization 5; the official client skips unstable layouts, so these are layouts
0-9 except where a layout was abandoned).  Variant names with an ``@label`` are matched by their task config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metiswam4d.eval.rdj_campaign import DIM_ORDER, aggregate, fmt, read_details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ref = json.loads((args.reference / "campaign.json").read_text())
    episodes = {v["name"]: int(v["episodes"]) for v in ref["variants"]}
    details, origin = {}, {}
    for spec in args.source:
        path, _, tasks = spec.partition(":")
        out = Path(path)
        camp = json.loads((out / "campaign.json").read_text())
        wanted = set(tasks.split(",")) if tasks else None
        for v in camp["variants"]:
            config = v["name"].split("@")[0]
            if config not in episodes or (wanted is not None and config not in wanted):
                continue
            rows = sorted(read_details(out, v["name"]), key=lambda r: int(r["layout_id"]))[:episodes[config]]
            if len(rows) != episodes[config]:
                raise ValueError(f"{out}: {v['name']} has {len(rows)} protocol episodes, expected {episodes[config]}")
            details[config], origin[config] = rows, f"{out.parent.name}/{out.name}"
    missing = sorted(set(episodes) - set(details))
    if missing:
        raise ValueError(f"no source for {missing}")
    agg = aggregate(ref, details, None)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["| Dimension | SR | Score |", "|---|---:|---:|"]
    for dim in ("Overall",) + tuple(DIM_ORDER) + ("Gen-Std", "Gen-Random"):
        d = agg["overall"] if dim == "Overall" else agg["dimensions"].get(dim, {})
        lines.append(f"| {dim} | {fmt(d.get('sr'))} | {fmt(d.get('score'))} |")
    lines += ["", f"Successes {agg['successes']} / {agg['episodes']}", "",
              "| Task config | Dimension | Successes | Source |", "|---|---|---:|---|"]
    for v in ref["variants"]:
        r = agg["variants"][v["name"]]
        lines.append(f"| {v['name']} | {v['dimension']} | {r['success']} / {r['n']} | {origin[v['name']]} |")
    (args.out / "fusion_report.md").write_text("\n".join(lines) + "\n")
    (args.out / "fusion_report.json").write_text(json.dumps({"aggregate": agg, "origin": origin,
                                                             "sources": args.source}, indent=1, default=str))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
