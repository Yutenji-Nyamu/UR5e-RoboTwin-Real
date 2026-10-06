"""Assemble the Memory / Open training root (``IW0EpisodeDataset``): official demos of the six Memory tasks + teacher
rollouts with ground-truth 4D Track.

    <root>/<task>/train_4d        -> symlink: object-overlay episodes (RDJ_selfplay/object_track_v2) where they exist,
                                     else the RDJ_MetisWAM4D episodes (robot-only Track)
    <root>/<task>/rollout_<tag>/  teacher episodes written in place by ``build_gt4d.py``
    index.jsonl                   official train / val rows of the six Memory tasks + ``--repeat`` rows per teacher episode
    instructions.jsonl            {"key": "<task>/<variant>/episodeN", "instructions": [...]}
    text_cache/                   UMT5 features (compact_prompt_sqlite_v1) of every templated instruction

    PYTHONPATH=. /usr/bin/python3.10 metiswam4d_inspired_by_internw0/data_prep/assemble_dataset.py --root <root> \
        --tags t1 --repeat 2 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3

import torch

from metiswam4d.data.rt2.text_cache import format_prompt

RDJ = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D")
OBJECTS = Path("/ytech_milm_intern/danglingwei/datas/RDJ_selfplay/object_track_v3_shift1")
UMT5 = "/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MEMORY_TASKS = ("cover_blocks", "match_and_pick_from_conveyor", "swap_blocks", "swap_T", "press_by_number",
                "imitate_sorting_sequence")


def official_instructions() -> dict[str, list[str]]:
    out = {}
    for line in (RDJ / "episode_instructions_official.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            parts = Path(row["source"]).parts
            out[f"{parts[-4]}/{parts[-3]}/{Path(parts[-1]).stem}"] = list(row["instructions"])
    return out


def link_official(root: Path) -> list[dict]:
    rows = [json.loads(l) for l in (RDJ / "index.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["task"] in MEMORY_TASKS]
    for task in MEMORY_TASKS:
        (root / task).mkdir(parents=True, exist_ok=True)
        link = root / task / "train_4d"
        src = OBJECTS / task / "train_4d"
        episodes = [r["episode"] for r in rows if r["task"] == task]
        if not (src.is_dir() and all((src / f"episode{e}" / "track4d.h5").exists() for e in episodes)):
            src = RDJ / task / "train_4d"
        if link.is_symlink() and os.readlink(link) != str(src):
            link.unlink()
        if not link.is_symlink():
            os.symlink(src, link)
        print(f"{task}: {len(episodes)} official episodes -> {os.readlink(link)}", flush=True)
    return rows


def teacher_rows(root: Path, tags: list[str], val_per_task: int) -> tuple[list[dict], dict[str, list[str]]]:
    """Teacher episodes; the ``val_per_task`` highest-numbered episodes of every task are held out."""
    rows, instr = [], {}
    for tag in tags:
        for meta_path in sorted(root.glob(f"*/rollout_{tag}/episode*/meta.json")):
            meta = json.loads(meta_path.read_text())
            if meta.get("status") != "ok" or not meta.get("schema", "").endswith("gt4d.v1"):
                continue
            d = meta_path.parent
            rows.append({"task": d.parent.parent.name, "variant": d.parent.name, "episode": int(d.name[7:]),
                         "frames": int(meta["frames"]), "split": "train"})
            instr[f"{d.parent.parent.name}/{d.parent.name}/{d.name}"] = [meta["instruction"]]
    for task in {r["task"] for r in rows}:
        for r in sorted((r for r in rows if r["task"] == task), key=lambda r: r["episode"])[-val_per_task:]:
            r["split"] = "val"
    return rows, instr


def build_text_cache(out: Path, prompts: list[str], device: str) -> int:
    out.mkdir(exist_ok=True)
    db = sqlite3.connect(out / "embeddings.sqlite3", timeout=120)
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (prompt TEXT PRIMARY KEY, length INTEGER NOT NULL, "
               "value BLOB NOT NULL) WITHOUT ROWID")
    done = {row[0] for row in db.execute("SELECT prompt FROM embeddings")}
    todo = [q for q in prompts if q not in done]
    if todo:
        from metiswam4d.data.human.text import UMT5Online
        encoder = UMT5Online(UMT5, torch.device(device))
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
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
                                                   "dtype": "bfloat16", "format": "compact_prompt_sqlite_v1"}, indent=2))
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tags", default="", help="comma-separated teacher rollout tags")
    ap.add_argument("--repeat", type=int, default=1, help="index rows per teacher episode")
    ap.add_argument("--task-repeat", default="", help="per-task override of --repeat, e.g. general_pickup=4,...")
    ap.add_argument("--teacher-val", type=int, default=3, help="held-out teacher episodes per task")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    official = link_official(root)
    instr_all = official_instructions()
    instructions = {f"{r['task']}/{r['variant']}/episode{r['episode']}": instr_all[f"{r['task']}/{r['variant']}/episode{r['episode']}"]
                    for r in official}
    teacher, teacher_instr = teacher_rows(root, [t for t in args.tags.split(",") if t], args.teacher_val)
    instructions.update(teacher_instr)
    repeat = {k: int(v) for k, v in (kv.split("=") for kv in args.task_repeat.split(",") if kv)}
    with open(root / "index.jsonl.tmp", "w") as f:
        for r in official:
            f.write(json.dumps(r) + "\n")
        for r in teacher:
            for _ in range(repeat.get(r["task"], args.repeat) if r["split"] == "train" else 1):
                f.write(json.dumps(r) + "\n")
    os.replace(root / "index.jsonl.tmp", root / "index.jsonl")
    with open(root / "instructions.jsonl", "w") as f:
        for key, values in sorted(instructions.items()):
            f.write(json.dumps({"key": key, "instructions": values}) + "\n")
    prompts = sorted({format_prompt(t) for values in instructions.values() for t in values})
    encoded = build_text_cache(root / "text_cache", prompts, args.device)
    per_task: dict[str, list[int]] = {}
    for r in official + teacher:
        per_task.setdefault(r["task"], [0, 0, 0])
        per_task[r["task"]][0 if r["variant"] == "train_4d" and r["split"] == "train" else
                            1 if r["split"] == "val" else 2] += 1
    summary = {"official_rows": len(official), "teacher_episodes": len(teacher), "repeat": args.repeat,
               "task_repeat": repeat,
               "prompts": len(prompts), "newly_encoded": encoded,
               "per_task_official_train_val_teacher": per_task}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
