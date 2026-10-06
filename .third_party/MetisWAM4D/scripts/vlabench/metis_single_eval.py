"""The official OpenWAM VLABench client (``benchmarks/vlabench/single_eval.py``, same arguments) for a MetisWAM4D
server (``metiswam4d/eval/vlabench_policy.py``).

The only change is the policy adapter: every observation message additionally carries the Track conditions the model
was trained with, computed from the evaluator's own ``obs`` exactly as ``scripts/vlabench/generate_4d.py`` recorded them
for training:
    head_track     ``forward`` RGB 480x480 -> 320x240 (LANCZOS), base64 PNG
    depth_mm_f16   ``forward`` z-depth sampled on the 240x320 stretched grid, millimetres, float16 bytes
    mask_u8        robot pixels on the same grid: segmentation type geom and geom id in the Franka's 0..72, uint8 bytes
Run with the VLABench venv; PYTHONPATH must contain the VLABench repo, ``benchmarks/vlabench`` and the client stubs.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import numpy as np

OPENWAM = Path(os.environ.get("OPENWAM_ROOT", "/m2v_intern_v3/danglingwei/codes/OpenWAM_Official_260913"))
sys.path.insert(0, str(OPENWAM / "benchmarks" / "vlabench"))
sys.path.insert(0, str(OPENWAM))

import single_eval  # noqa: E402
from benchmarks.utils import client as ow_client  # noqa: E402
from openwam2vlabench_interface import OpenWAMVLABenchPolicy  # noqa: E402
from PIL import Image  # noqa: E402

SRC = 480
GRID_H, GRID_W = 240, 320
ROWS = np.floor((np.arange(GRID_H) + 0.5) * SRC / GRID_H).astype(np.int64)
COLS = np.floor((np.arange(GRID_W) + 0.5) * SRC / GRID_W).astype(np.int64)
HEAD = 2                 # obs camera order: right, left, forward, wrist
ROBOT_GEOM_MAX = 72      # the Franka is the first model in every VLABench scene: geoms 0..72
MJ_OBJ_GEOM = 5


def track_conditions(obs: dict) -> dict:
    rgb = np.asarray(obs["rgb"])[HEAD].astype(np.uint8)
    head = np.asarray(Image.fromarray(rgb).resize((GRID_W, GRID_H), Image.LANCZOS), dtype=np.uint8)
    depth = np.asarray(obs["depth"])[HEAD][np.ix_(ROWS, COLS)].astype(np.float32) * 1000.0
    seg = np.asarray(obs["segmentation"])[HEAD][np.ix_(ROWS, COLS)]
    mask = (seg[..., 1] == MJ_OBJ_GEOM) & (seg[..., 0] >= 0) & (seg[..., 0] <= ROBOT_GEOM_MAX)
    return {"head_track": ow_client.encode_numpy_b64(head),
            "depth_mm_f16": base64.b64encode(depth.astype(np.float16).tobytes()).decode(),
            "mask_u8": base64.b64encode(mask.astype(np.uint8).tobytes()).decode()}


class _WithConditions:
    """Wraps the WebSocket client: ``predict`` payloads get the conditions of the observation being answered."""

    def __init__(self, inner, owner):
        self._inner, self._owner = inner, owner

    def predict(self, payload: dict) -> dict:
        return self._inner.predict({**payload, "metis": track_conditions(self._owner._obs)})

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MetisVLABenchPolicy(OpenWAMVLABenchPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = _WithConditions(self._client, self)
        self._obs = None

    @property
    def name(self) -> str:
        return "openwam"   # same result layout as the official client (``<save_dir>/<track>/openwam``)

    def predict(self, obs: dict, **kwargs):
        self._obs = obs
        return super().predict(obs, **kwargs)


single_eval.OpenWAMVLABenchPolicy = MetisVLABenchPolicy

if __name__ == "__main__":
    single_eval.main()
