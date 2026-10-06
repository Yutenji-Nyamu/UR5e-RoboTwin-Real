"""Throwaway: what does SAM3 detect on H13_L3 with generic prompts at a low threshold?"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_human_depth_masks as P  # noqa: E402

H = Path("/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/human/imitator_human_v1_levels")
S = Path("/dev/shm/diag")
S.mkdir(exist_ok=True)
panels = []
for d, prompts in [("human_H13_L3", ["juice box", "box", "bottle", "scanner", "barcode scanner", "handheld device", "drink"]),
                   ("human_H10_L3", ["bag of chips", "chips", "snack bag", "plastic bag", "bag", "package"])]:
    info = json.loads((H / d / "meta/info.json").read_text())
    ego = P.ego_key(info)
    eps = P.episodes_of(H / d, ego)
    clip = S / f"{d}.mp4"
    P.cut_episode(eps[0], clip)
    rgb = P.read_frames(clip)
    n, h, w = rgb.shape[:3]
    for thr in (0.3, 0.15):
        P.DET_SCORE_THRESH = thr
        det = P.sam_detect(clip, n, (h, w), prompts, frames=list(range(0, n, 4)))
        print(f"== {d} thr={thr}", flush=True)
        for p in prompts:
            hit = det[p].reshape(det[p].shape[0], -1).any(1)
            print(f"   {p:22s} fires {hit.sum():3d}/{len(hit)} frames, px/frame {det[p].sum() / max(hit.sum(), 1):6.0f}", flush=True)
    best = max(prompts, key=lambda p: det[p].sum())
    j = int(np.argmax(det[best].reshape(det[best].shape[0], -1).sum(1)))
    t = j * 4
    vis = rgb[t].copy()
    mm = det[best][j]
    vis[mm] = (0.3 * vis[mm] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.putText(vis, f"{d} {best!r} f{t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    panels.append(np.concatenate([rgb[t], vis], 1))
cv2.imwrite("/dev/shm/diag/levels_diag.png", cv2.cvtColor(np.concatenate(panels, 0), cv2.COLOR_RGB2BGR))
print("saved", flush=True)
