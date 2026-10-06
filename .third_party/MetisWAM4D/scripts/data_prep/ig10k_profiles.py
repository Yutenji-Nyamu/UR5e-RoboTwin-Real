"""Which IG-10K subsets a preprocessing run covers, and where its outputs live.

The human and robot subsets share one pipeline (resize -> DA3 depth + masks -> RAFT Track4D) but
differ in what shipped with them and therefore in what has to be estimated:

  human   imitator_human_v1 (+ _levels)   MANO hands shipped -> arm never segmented on v1;
                                          _levels: SAM 3.1 for both arm and objects
  robot   imitator_robot_v1               object instance masks shipped, no arm/hand annotation
                                          -> SAM 3.1 detects the robot arm on every frame and it is
                                          written into the shipped id map under ARM_ID.
                                          Only the 210 dirs listed in robot_desc.json are used; the
                                          `_e` / `_e_d` dirs are the paper's few-shot split.

Every script takes `--profile {human,robot}`; the default stays `human` so the finished human set
is untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

RAW_ROOT = Path("/ytech_milm_intern/danglingwei/datas/IG-10K-Dataset")
PREPROCESSED = Path("/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed")

# instance id reserved for the SAM-detected arm inside a shipped instance-id map (shipped ids are
# small consecutive integers starting at 1)
ARM_ID = 200

PROFILES = {
    "human": {
        "subsets": ("imitator_human_v1", "imitator_human_v1_levels"),
        "out_root": PREPROCESSED / "human",
        "arm_prompts": ("human arm",),
        "desc": RAW_ROOT / "_meta_extra/task_desc/human_desc.json",
        "include": None,
        "det_thresh": 0.3,
    },
    "robot": {
        "subsets": ("imitator_robot_v1",),
        "out_root": PREPROCESSED / "robot",
        # single prompt, per-frame detection (same recipe as "human arm" on _levels); covers both
        # Realman arms and their grippers in the demo overlays
        "arm_prompts": ("robot arm",),
        "desc": RAW_ROOT / "_meta_extra/task_desc/robot_desc.json",
        "include": "desc",  # only task dirs that have an entry in robot_desc.json
        # 48 robot dirs ship no mask stream and go through SAM; screen at the threshold the
        # _levels repair settled on (0.3 lost small objects)
        "det_thresh": 0.15,
    },
}


def profile(name: str) -> dict:
    p = dict(PROFILES[name])
    p["name"] = name
    p["raw_root"] = RAW_ROOT
    if p["include"] == "desc":
        p["include"] = set(json.loads(Path(p["desc"]).read_text()).keys())
    return p


def allowed(p: dict, task_dir_name: str) -> bool:
    return p["include"] is None or task_dir_name in p["include"]
