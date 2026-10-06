#!/usr/bin/env python
"""Spot-check videos for the Kling hand Track4D: sample clips from the hand-track subset, render
RGB | hand-track side by side from the saved MANO fits (mano_hi), write mp4s + a contact sheet.

    /usr/bin/python3.10 kling_review_hand_track.py --n 60
"""
import argparse
import io
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kling_hand_mano_track as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = torch.device("cuda")
OUT = K.OUT_ROOT / "review_hand_track"
OUT.mkdir(exist_ok=True)
scale = np.asarray(json.load(open(K.OUT_ROOT / "track_hi" / "codec.json"))["scale_xyz_metres"], np.float32)
t = pq.read_table(K.OUT_ROOT / "KlingHumanEgo20M_30fps_hi_handtrack_subset.parquet")
rng = np.random.default_rng(a.seed)
rows = t.take(rng.choice(t.num_rows, a.n, replace=False)).to_pylist()
sheet = []
tars = {}
for i, r in enumerate(rows):
    fh = tars.get(r["mano_path"]) or tars.setdefault(r["mano_path"], open(r["mano_path"], "rb"))
    fh.seek(r["mano_offset"])
    fit = dict(np.load(io.BytesIO(fh.read(r["mano_size"]))))
    frames = K.render_clip(fit, r, scale, dev)
    with open(r["video_path"], "rb") as v:
        v.seek(r["offset"])
        Path("/dev/shm/kr_rgb.mp4").write_bytes(v.read(r["size"]))
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "/dev/shm/kr_rgb.mp4", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       capture_output=True)
    rgb = np.frombuffer(p.stdout, np.uint8).reshape(-1, r["height"], r["width"], 3)
    n = min(len(rgb), len(frames))
    side = np.concatenate([rgb[:n], frames[:n]], 2)
    key = r["blobstore_key"].split("-")[-1][:12]
    out = OUT / f"{i:03d}_{key}_cov{r['hand_det_per_frame']:.2f}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{side.shape[2]}x{side.shape[1]}", "-r", "30", "-i", "pipe:",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
                   input=side.tobytes(), capture_output=True, check=True)
    per = frames[:n].reshape(n, -1, 3).any(-1).mean(1)
    k = int(np.argmax(per))
    tile = side[k].copy()
    cv2.putText(tile, f"{i:03d} cov={r['hand_det_per_frame']:.2f} score={r['ego_mani_score']:.2f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    sheet.append(tile)
    print(f"{out.name}: {n} frames, hand in {(per > 0).mean() * 100:.0f}% frames", flush=True)
rows_per_sheet = 10
for s in range(0, len(sheet), rows_per_sheet):
    tiles = sheet[s:s + rows_per_sheet]
    w = max(x.shape[1] for x in tiles)
    tiles = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in tiles]
    cv2.imwrite(str(OUT / f"contact_sheet_{s // rows_per_sheet:02d}.png"), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
print("written", OUT, flush=True)
