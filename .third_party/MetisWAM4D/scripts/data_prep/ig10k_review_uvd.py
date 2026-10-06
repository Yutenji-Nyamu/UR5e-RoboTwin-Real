#!/usr/bin/env python
"""Spot-check IG-10K episodes in the Δuvd track representation the trainer uses.

For each sampled episode: RGB | mask overlay (arm red, objects green) | uvd track, one video frame per
t -> t+4 pair, encoded exactly like `metiswam4d/data/human/ig10k.py` (uvd_from_xyz -> soft-shrink ->
encode_uvd, mu=31, invalid black, zero = 128 grey).  Also a contact sheet (3 frames per episode) and a
6-consecutive-frame strip per episode to judge flicker.  CPU only.

    /usr/bin/python3.10 scripts/data_prep/ig10k_review_uvd.py --profile robot \
        --dirs robot_H10_L0,robot_H1_L2,robot_H36_L1,robot_H28_L0,robot_H5_L1,robot_H42_L3 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ig10k_human_depth_masks as DM  # noqa: E402
import ig10k_render_track_rgb as R  # noqa: E402
from ig10k_anno_reader import IG10KAnno  # noqa: E402
from metiswam4d.data.rt2.codec import CODEC_ID, SHRINK_D_M, encode_uvd, uvd_from_xyz  # noqa: E402


def overlay(rgb, objects, arm):
    out = rgb.copy()
    col = np.zeros_like(rgb)
    col[objects] = (0, 255, 0)
    if arm is not None:
        col[arm] = (255, 0, 0)
    m = col.any(-1)
    out[m] = (0.5 * out[m] + 0.5 * col[m]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("human", "robot"), default="robot")
    ap.add_argument("--dirs", default="robot_H10_L0,robot_H1_L2,robot_H36_L1,robot_H28_L0,robot_H5_L1,robot_H42_L3")
    ap.add_argument("--per-dir", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    DM.apply_profile(a.profile)
    anno = IG10KAnno(profile=a.profile)
    out_dir = Path(a.out) if a.out else anno.anno_root / "review_uvd"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tau_d = SHRINK_D_M["ig10k"]
    sheet, stats = [], []
    for d in a.dirs.split(","):
        sub = next(s for s in anno.subsets if (anno.anno_root / s / d).is_dir())
        td = anno.anno_root / sub / d
        info = json.loads((td / "meta" / "info.json").read_text())
        ego = DM.ego_key(info)
        rows = {r["episode"]: r for r in DM.episodes_of(td, ego)}
        eps = anno.episode_ids(sub, d)
        for ep in rng.choice(eps, size=min(a.per_dir, len(eps)), replace=False):
            ep = int(ep)
            e = anno.load(sub, d, ep, with_depth=True)
            tr = anno.load_track4d(sub, d, ep)
            src = tr["src"]
            fg = tr["valid"] & (e.objects[src] | (e.arm[src] if e.arm is not None else False))
            uvd = uvd_from_xyz(tr["delta"], e.depth[src], tr["K"])            # (P,H,W,3) px, px, m
            track_px, _ = encode_uvd(uvd, fg, e.objects.shape[-1], tau_d)      # uint8 (P,H,W,3)
            clip = Path("/dev/shm") / f"uvd_{d}_{ep}.mp4"
            try:
                DM.cut_episode(rows[ep], clip)
                rgb = DM.read_frames(clip)
            finally:
                clip.unlink(missing_ok=True)
            P = len(src)
            ov = np.stack([overlay(rgb[t], e.objects[t], None if e.arm is None else e.arm[t]) for t in src])
            side = np.concatenate([rgb[src], ov, track_px], 2)
            out = out_dir / f"{d}_ep{ep:04d}.mp4"
            R.write_mp4(side, out, pix_fmt="yuv420p", crf=18)
            # statistics on the painted pixels
            mag_uv = np.linalg.norm(uvd[..., :2], axis=-1)[fg]
            dd = np.abs(uvd[..., 2][fg]) * 1000
            grey = np.abs(track_px.astype(np.int16) - 128)[fg]
            static = fg & (np.linalg.norm(uvd[..., :2], axis=-1) < 0.5)
            st = {"dir": d, "ep": ep, "pairs": P, "fg_px_per_pair": float(fg.sum() / P),
                  "uv_px_p50_p99": [float(np.percentile(mag_uv, 50)), float(np.percentile(mag_uv, 99))],
                  "dd_mm_p50_p99": [float(np.percentile(dd, 50)), float(np.percentile(dd, 99))],
                  "static_frac": float(static.sum() / max(fg.sum(), 1)),
                  "static_grey_dev_p50": float(np.median(np.abs(track_px.astype(np.int16) - 128)[static])) if static.any() else 0.0,
                  "moving_grey_dev_p50": float(np.median(grey[mag_uv >= 3])) if (mag_uv >= 3).any() else 0.0}
            stats.append(st)
            print(f"{out.name}: {P} pairs, fg {st['fg_px_per_pair']:.0f} px/pair, |uv| p50/p99 {st['uv_px_p50_p99'][0]:.2f}/"
                  f"{st['uv_px_p50_p99'][1]:.1f} px, |dd| p50/p99 {st['dd_mm_p50_p99'][0]:.1f}/{st['dd_mm_p50_p99'][1]:.0f} mm, "
                  f"static {st['static_frac']:.2f} grey-dev {st['static_grey_dev_p50']:.0f}, moving grey-dev {st['moving_grey_dev_p50']:.0f}",
                  flush=True)
            # contact sheet: 3 frames spread over the episode; strip: 6 consecutive pairs around peak motion
            for k in np.linspace(P * 0.15, P * 0.85, 3).round().astype(int):
                tile = side[k].copy()
                cv2.putText(tile, f"{d} ep{ep} pair{k}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                sheet.append(tile)
            peak = int(np.argmax([(np.linalg.norm(uvd[j, ..., :2], axis=-1) * fg[j]).sum() for j in range(P)]))
            k0 = int(np.clip(peak - 3, 0, max(P - 6, 0)))
            strip = np.concatenate([track_px[k] for k in range(k0, min(k0 + 6, P))], 1)
            strip = np.concatenate([np.concatenate([rgb[src[k]] for k in range(k0, min(k0 + 6, P))], 1), strip], 0)
            cv2.imwrite(str(out_dir / f"{d}_ep{ep:04d}_strip.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "contact_sheet.png"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR))
    (out_dir / "stats.json").write_text(json.dumps({"codec": CODEC_ID, "tau_d_m": tau_d, "episodes": stats}, indent=1))
    print("written", out_dir, flush=True)


if __name__ == "__main__":
    main()
