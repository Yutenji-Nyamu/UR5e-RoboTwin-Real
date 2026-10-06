#!/usr/bin/env python
"""Render a few Kling clips side by side (RGB | MANO hand Track4D) for eyeballing before batch."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kling_hand_mano_track as K  # noqa: E402

N_DEMOS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
dev = torch.device("cuda")
D = Path("/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M/demo_hand_track")
D.mkdir(exist_ok=True)
scale = np.asarray(json.load(open(K.OUT_ROOT / "track_hi" / "codec.json"))["scale_xyz_metres"], np.float32)
t = pq.read_table(K.SRC_PARQUET, columns=["blobstore_key", "video_path", "offset", "size", "nb_frames", "width",
                                           "height", "src_fps", "src_nb_frames", "wilor_npz_archive",
                                           "wilor_npz_member", "ego_mani_score", "caption"])
cands = [r for r in t.slice(0, 3000).to_pylist()
         if r["width"] == 512 and r["nb_frames"] >= 90 and r["ego_mani_score"] > 0.8]
pick = [cands[int(i)] for i in np.linspace(0, len(cands) - 1, N_DEMOS)]
sheet = []
for n, r in enumerate(pick):
    fit = K.fit_clip(r, dev)
    frames = K.render_clip(fit, r, scale, dev)
    with open(r["video_path"], "rb") as fh:
        fh.seek(r["offset"])
        Path("/dev/shm/rgb.mp4").write_bytes(fh.read(r["size"]))
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "/dev/shm/rgb.mp4", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       capture_output=True)
    rgb = np.frombuffer(p.stdout, np.uint8).reshape(-1, r["height"], r["width"], 3)[:len(frames)]
    side = np.concatenate([rgb, frames[:len(rgb)]], 2)
    key = r["blobstore_key"].split("-")[-1][:12]
    out = D / f"demo{n}_{key}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{side.shape[2]}x{side.shape[1]}", "-r", "30", "-i", "pipe:",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
                   input=side.tobytes(), capture_output=True, check=True)
    per = frames.reshape(len(frames), -1, 3).any(-1).mean(1)
    k = int(np.argmax(per))
    sheet.append(side[k])
    cap = json.loads(r["caption"])["global"][:80]
    print(f"demo{n}: {len(frames)} frames, hands in {int((per > 0).sum())} frames, "
          f"joint RMSE {np.median(fit['joint_rmse_m']) * 1000:.1f} mm -> {out.name}\n     {cap}", flush=True)
cv2.imwrite(str(D / "contact_sheet.png"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR))
print("written to", D)
