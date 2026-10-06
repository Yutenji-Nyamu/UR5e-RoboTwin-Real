"""Read the offline UMT5 cache written by the predecessor (format ``compact_prompt_sqlite_v1``).

Prompts are stored after the RoboTwin training template; the same template must be applied to a raw
instruction before lookup.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch import Tensor

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: "


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE + instruction


class PromptTextCache:
    def __init__(self, root: str | Path, width: int = 4096):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != "compact_prompt_sqlite_v1" or int(manifest.get("width", width)) != width:
            raise ValueError(f"unexpected text cache manifest: {manifest}")
        self.width = width
        self._conn: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:  # lazily per worker process; read-only + immutable, so sharing it across threads is safe
            self._conn = sqlite3.connect(f"file:{self.root / 'embeddings.sqlite3'}?mode=ro&immutable=1", uri=True,
                                         check_same_thread=False)
        return self._conn

    def __contains__(self, prompt: str) -> bool:
        return self._connection().execute("SELECT 1 FROM embeddings WHERE prompt=?", (prompt,)).fetchone() is not None

    def load(self, prompt: str) -> Tensor:
        row = self._connection().execute("SELECT length, value FROM embeddings WHERE prompt=?", (prompt,)).fetchone()
        if row is None:
            raise KeyError(f"prompt missing from the UMT5 cache: {prompt!r}")
        length, blob = row
        array = np.frombuffer(blob, dtype=np.uint16).copy().reshape(int(length), self.width)
        return torch.from_numpy(array).view(torch.bfloat16)

    def __getstate__(self):
        return {"root": self.root, "width": self.width, "_conn": None}

    def __setstate__(self, state):
        self.__dict__.update(state)


def load_instruction_index(path: str | Path) -> dict[str, list[str]]:
    """``episode_instructions_sim_aligned.jsonl`` -> {"<task>/<variant>/episodeN": [prompts]}."""
    index: dict[str, list[str]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parts = Path(row["source"]).parts
            # .../GeoRobotwin/<task>/<variant>/data/episodeN.hdf5
            task, variant, episode = parts[-4], parts[-3], Path(parts[-1]).stem
            pool = row.get("seen") or row.get("unseen") or []
            index[f"{task}/{variant}/{episode}"] = [format_prompt(p) for p in pool]
    return index


__all__ = ["PROMPT_TEMPLATE", "PromptTextCache", "format_prompt", "load_instruction_index"]
