"""QC panel for scripts/ig10k/sim_gt4d.py episodes: RGB | part overlay | (du, dv) arrows at three times.

    /usr/bin/python3.10 scripts/ig10k/qc_sim_gt4d.py <episode.h5> [<episode.h5> ...] --out panel.png
"""
import argparse
import json

import cv2
import h5py
import numpy as np

PART_COLORS = {1: (60, 120, 255), 2: (0, 220, 220), 3: (255, 50, 50)}


def panel(path: str) -> np.ndarray:
    with h5py.File(path, "r") as h:
        n = h.attrs["frames"]
        ts = [0, (n - 4) // 3, 2 * (n - 4) // 3]
        rows = []
        for t in ts:
            rgb = cv2.imdecode(h["rgb"][t], cv2.IMREAD_COLOR)[..., ::-1]
            part = h["part"][t]
            d = h["delta_uvd"][min(t, n - 5)].astype(np.float32)
            H, W = part.shape
            base = cv2.resize(rgb, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
            over = base.copy()
            big = cv2.resize(part, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
            for k, c in PART_COLORS.items():
                over[big == k] = (0.45 * over[big == k] + 0.55 * np.array(c)).astype(np.uint8)
            arrows = base.copy() // 2 + 64
            step = 6
            for i in range(0, H, step):
                for j in range(0, W, step):
                    if part[i, j] == 0:
                        continue
                    du, dv = d[i, j, 0], d[i, j, 1]
                    if np.hypot(du, dv) < 0.3:
                        continue
                    c = PART_COLORS[int(part[i, j])]
                    p0 = (2 * j + 1, 2 * i + 1)
                    p1 = (int(p0[0] + 2 * 3 * du), int(p0[1] + 2 * 3 * dv))
                    cv2.arrowedLine(arrows, p0, p1, c, 1, tipLength=0.3)
            row = np.concatenate([base, over, arrows], 1)
            cv2.putText(row, f"{h.attrs['env_dir']} ep{h.attrs['episode']} t={t}/{n}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            rows.append(row)
        moving = [e for e, p in zip(json.loads(h.attrs["entities"]), json.loads(h.attrs["entity_part"])) if p == 3]
        cv2.putText(rows[-1], "moving: " + ", ".join(moving)[:90], (6, rows[-1].shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return np.concatenate(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    img = np.concatenate([panel(p) for p in args.episodes], 0)
    cv2.imwrite(args.out, img[..., ::-1])
    print("wrote", args.out, img.shape)


if __name__ == "__main__":
    main()
