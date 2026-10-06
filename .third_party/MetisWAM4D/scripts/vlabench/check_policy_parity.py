"""Open-loop check of the VLABench evaluation policy on held-out generated episodes.

Each window is built the way ``scripts/vlabench/metis_single_eval.py`` builds the observation message (views LANCZOS
480 -> slot size, head 480 -> 240x320, depth / robot mask on the 240x320 grid, raw EEF10 from ``state7``) and passed
through ``VLABenchServer.request`` decoding and ``VLABenchPolicy.predict``; the 32 predicted EEF10 actions are compared
with the episode's ``actions7`` converted the same way.  A broken normalisation, layout or condition path shows up as a
position error of several centimetres instead of a few millimetres.

    PYTHONPATH=. /usr/bin/python3.10 scripts/vlabench/check_policy_parity.py --checkpoint <model_bf16.pt>
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from metiswam4d.data.vlabench.episode_dataset import euler7_to_eef10
from metiswam4d.eval.vlabench_policy import VLABenchPolicy, VLABenchServer

ROOT = Path("/ytech_milm_intern/danglingwei/datas/VLABench/gen4d")
SLOTS = {"head_rgb": (320, 256), "left_rgb": (160, 128), "right_rgb": (160, 128)}


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()


def message(h: h5py.File, s: int) -> dict:
    frames = {k: Image.open(io.BytesIO(h[k][s].tobytes())).convert("RGB") for k in SLOTS}
    views = {k: np.asarray(frames[k].resize(size, Image.LANCZOS), np.uint8) for k, size in SLOTS.items()}
    head = np.asarray(frames["head_rgb"].resize((320, 240), Image.LANCZOS), np.uint8)
    depth = np.asarray(h["head_depth_mm"][s], np.float16)
    mask = (np.asarray(h["role"][s]) == 1).astype(np.uint8)
    return {"images": {"head_camera": png_b64(views["head_rgb"]), "left_wrist_camera": png_b64(views["left_rgb"]),
                       "right_wrist_camera": png_b64(views["right_rgb"])},
            "prompt": str(h.attrs["instruction"]),
            "state": euler7_to_eef10(np.asarray(h["state7"][s], np.float32)).tolist(),
            "metis": {"head_track": png_b64(head), "depth_mm_f16": base64.b64encode(depth.tobytes()).decode(),
                      "mask_u8": base64.b64encode(mask.tobytes()).decode()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/stage3_vlabench_v1.yaml")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    policy = VLABenchPolicy(args.config, args.checkpoint)
    server = VLABenchServer(policy)
    server.reset()
    rows = [json.loads(l) for l in open(ROOT / "index.jsonl") if json.loads(l)["split"] == "val"]
    rng = np.random.default_rng(0)
    results = []
    for row in [rows[i] for i in rng.choice(len(rows), args.windows, replace=False)]:
        with h5py.File(ROOT / row["task"] / f"episode{row['episode']}.h5") as h:
            n = int(h.attrs["frames"])
            s = int(rng.integers(0, n - 33))
            req = server.request(message(h, s))
            gt = euler7_to_eef10(np.asarray(h["actions7"][s:s + 32], np.float32))
        pred = policy.predict(req)["eef10"]
        pos = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1)
        grip = float((np.round(pred[:, 9]) != gt[:, 9]).mean())
        rot = float(np.abs(pred[:, 3:9] - gt[:, 3:9]).mean())
        results.append({"task": row["task"], "episode": row["episode"], "s": s, "pos_mean_cm": float(pos.mean() * 100),
                        "pos_max_cm": float(pos.max() * 100), "rot6d_abs": rot, "gripper_mismatch": grip})
        print(json.dumps(results[-1]), flush=True)
    summary = {k: float(np.mean([r[k] for r in results])) for k in ("pos_mean_cm", "pos_max_cm", "rot6d_abs",
                                                                     "gripper_mismatch")}
    print("SUMMARY", json.dumps(summary), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "windows": results}, indent=1))


if __name__ == "__main__":
    main()
