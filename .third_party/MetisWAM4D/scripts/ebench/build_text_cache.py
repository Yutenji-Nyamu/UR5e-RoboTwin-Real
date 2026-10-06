"""UMT5 feature cache (compact_prompt_sqlite_v1) for every EBench instruction, in the Alpha deploy template.

    PYTHONPATH=. /usr/bin/python3.10 scripts/ebench/build_text_cache.py --device cuda:0
"""
import argparse
import json
from pathlib import Path
import sqlite3

import torch

from metiswam4d.data.human.text import UMT5Online
from metiswam4d.data.rt2.text_cache import format_prompt

ROOT = Path("/ytech_milm_intern/danglingwei/datas/EBench/EBench-Dataset")
UMT5 = "/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def collect_prompts(root: Path) -> list[str]:
    prompts = set()
    for meta in sorted(root.glob("*/*/meta")):
        for line in (meta / "tasks.jsonl").read_text().splitlines():
            if line.strip():
                prompts.add(format_prompt(json.loads(line)["task"]))
        for line in (meta / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                prompts.update(format_prompt(t) for t in json.loads(line).get("tasks") or [] if str(t).strip())
    return sorted(prompts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    prompts = collect_prompts(ROOT)
    out = ROOT / "text_cache"
    out.mkdir(exist_ok=True)
    db = sqlite3.connect(out / "embeddings.sqlite3", timeout=120)
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (prompt TEXT PRIMARY KEY, length INTEGER NOT NULL, "
               "value BLOB NOT NULL) WITHOUT ROWID")
    done = {row[0] for row in db.execute("SELECT prompt FROM embeddings")}
    todo = [q for q in prompts if q not in done]
    print(f"{len(prompts)} prompts, {len(todo)} to encode", flush=True)
    encoder = UMT5Online(UMT5, torch.device(args.device))
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        hidden, mask = encoder(chunk)
        records = []
        for q, h, m in zip(chunk, hidden, mask):
            e = h[: int(m.sum())].to("cpu", torch.bfloat16).contiguous()
            records.append((q, e.shape[0], e.view(torch.uint16).numpy().tobytes()))
        db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", records)
        db.commit()
    count = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    db.close()
    if count < len(prompts):
        raise SystemExit(f"text cache incomplete: {count} / {len(prompts)}")
    (out / "manifest.json").write_text(json.dumps({"checkpoint": UMT5, "count": count, "width": 4096,
                                                   "dtype": "bfloat16", "format": "compact_prompt_sqlite_v1"},
                                                  indent=2) + "\n")
    print(f"text cache: {count} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()
