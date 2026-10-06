"""UMT5 feature cache (compact_prompt_sqlite_v1) for every instruction in the generated VLABench episodes, in the Alpha
deploy template.  Re-runnable: only prompts not yet in the cache are encoded.

    PYTHONPATH=. /usr/bin/python3.10 scripts/vlabench/build_text_cache.py --device cuda:0
"""
import argparse
import json
from pathlib import Path
import sqlite3

import h5py
import torch

from metiswam4d.data.human.text import UMT5Online
from metiswam4d.data.rt2.text_cache import format_prompt

UMT5 = "/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/ytech_milm_intern/danglingwei/datas/VLABench/gen4d")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    root = Path(args.root)
    prompts = set()
    for path in sorted(root.glob("*/episode*.h5")):
        with h5py.File(path) as h:
            prompts.add(format_prompt(str(h.attrs["instruction"])))
    prompts = sorted(prompts)
    out = root / "text_cache"
    out.mkdir(exist_ok=True)
    db = sqlite3.connect(out / "embeddings.sqlite3", timeout=120)
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (prompt TEXT PRIMARY KEY, length INTEGER NOT NULL, "
               "value BLOB NOT NULL) WITHOUT ROWID")
    done = {row[0] for row in db.execute("SELECT prompt FROM embeddings")}
    todo = [q for q in prompts if q not in done]
    print(f"{len(prompts)} prompts, {len(todo)} to encode", flush=True)
    if todo:
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
    (out / "manifest.json").write_text(json.dumps({"checkpoint": UMT5, "count": count, "width": 4096,
                                                   "dtype": "bfloat16", "format": "compact_prompt_sqlite_v1"},
                                                  indent=2) + "\n")
    print(f"text cache: {count} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()
