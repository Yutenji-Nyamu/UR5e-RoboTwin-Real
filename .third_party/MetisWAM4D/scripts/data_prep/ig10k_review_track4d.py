#!/usr/bin/env python
"""Spot-check videos for IG-10K Track4D: a few episodes, RGB | mask overlay | track side by side,
track encoded with the frozen trainer codec. CPU only.

    /usr/bin/python3.10 ig10k_review_track4d.py --per-dir 1 --dirs human_H1,human_H7,human_H14,human_H33,human_H58,human_H10_L3,human_H14_L1,human_H11_L2
    /usr/bin/python3.10 ig10k_review_track4d.py --profile robot --dirs robot_H10_L0,robot_H1_L2 --per-dir 2

Overlay colours: arm (SAM) red, manipulated objects green, MANO hand track footprint blue.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_human_depth_masks as DM  # noqa: E402
import ig10k_render_track_rgb as R  # noqa: E402
from ig10k_anno_reader import IG10KAnno  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--profile", choices=("human", "robot"), default="human")
ap.add_argument("--dirs", default="human_H1,human_H7,human_H14,human_H33,human_H58,human_H10_L3,human_H14_L1,human_H11_L2")
ap.add_argument("--per-dir", type=int, default=1)
ap.add_argument("--masks-only", action="store_true", help="no track4d.h5 yet: RGB | mask overlay")
a = ap.parse_args()

DM.apply_profile(a.profile)
anno = IG10KAnno(profile=a.profile)
OUT = anno.anno_root / "review_track4d"
OUT.mkdir(exist_ok=True)


def overlay(rgb, objects, arm, hand=None):
    out = rgb.copy()
    col = np.zeros_like(rgb)
    col[objects] = (0, 255, 0)
    if arm is not None:
        col[arm] = (255, 0, 0)
    if hand is not None:
        col[hand] = (0, 128, 255)
    m = col.any(-1)
    out[m] = (0.5 * out[m] + 0.5 * col[m]).astype(np.uint8)
    return out


sheet = []
for d in a.dirs.split(","):
    sub = next(s for s in anno.subsets if (anno.anno_root / s / d).is_dir())
    td = anno.anno_root / sub / d
    info = json.loads((td / "meta" / "info.json").read_text())
    ego = DM.ego_key(info)
    rows = {r["episode"]: r for r in DM.episodes_of(td, ego)}
    for ep in anno.episode_ids(sub, d)[: a.per_dir]:
        e = anno.load(sub, d, ep, with_depth=False)
        clip = Path("/dev/shm") / f"rv_{d}_{ep}.mp4"
        try:
            DM.cut_episode(rows[ep], clip)
            rgb = DM.read_frames(clip)
        finally:
            clip.unlink(missing_ok=True)
        if a.masks_only:
            n = len(rgb)
            panels = [rgb, overlay(rgb, e.objects, e.arm)]
            tr = None
        else:
            tr = anno.load_track4d(sub, d, ep)
            rgb_track = R.encode(tr["delta"])
            rgb_track[~tr["valid"]] = 0
            n = len(rgb_track)
            hand = tr["hand"]
            panels = [rgb[:n], overlay(rgb[:n], e.objects[:n], None if e.arm is None else e.arm[:n], hand), rgb_track]
        side = np.concatenate(panels, 2)
        out = OUT / f"{d}_ep{ep:04d}.mp4"
        R.write_mp4(side, out, pix_fmt="yuv420p", crf=18)
        if tr is not None:
            k = int(np.argmax(tr["valid"].reshape(n, -1).sum(1)))
            stat = f"valid px/frame {tr['valid'].sum() / n:.0f}"
        else:
            k = int(np.argmax((e.objects | (e.arm if e.arm is not None else False)).reshape(n, -1).sum(1)))
            stat = f"obj px/frame {e.objects.sum() / n:.0f} arm px/frame {(e.arm.sum() / n) if e.arm is not None else -1:.0f}"
        tile = side[k].copy()
        cv2.putText(tile, f"{d} ep{ep} f{k}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        sheet.append(tile)
        print(f"{out.name}: {n} frames, {stat}", flush=True)
cv2.imwrite(str(OUT / "contact_sheet.png"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR))
print("written", OUT, flush=True)
