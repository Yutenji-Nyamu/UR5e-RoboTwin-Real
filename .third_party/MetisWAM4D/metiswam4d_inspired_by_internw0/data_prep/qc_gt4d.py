"""QC panel of a ground-truth 4D episode: head frames with role colours (arm blue, gripper cyan, object red, static
grey) and sparse delta-uv arrows (t -> t+4), plus the frame t+4 to compare against.

    PYTHONPATH=. /usr/bin/python3.10 metiswam4d_inspired_by_internw0/data_prep/qc_gt4d.py <episode dir> [<out png>]
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from metiswam4d.data.robodojo.episode_dataset import decode_bgr_jpeg

COLORS = {1: (40, 90, 255), 2: (255, 40, 40), 3: (0, 230, 230)}


def panel(ep: Path, out: Path, frames: int = 6) -> None:
    with h5py.File(ep / "source.hdf5") as src, h5py.File(ep / "track4d.h5") as t4d:
        n = t4d["role"].shape[0]
        ts = np.linspace(0, n - 5, frames).astype(int)
        rows = []
        for t in ts:
            rgb = np.asarray(decode_bgr_jpeg(src["observation/head_camera/rgb"][t]), np.uint8).copy()
            nxt = np.asarray(decode_bgr_jpeg(src["observation/head_camera/rgb"][t + 4]), np.uint8)
            role = t4d["role"][t]
            uvd = t4d["delta_uvd"][t].astype(np.float32)
            over = rgb.astype(np.float32)
            for r, c in COLORS.items():
                m = role == r
                over[m] = 0.45 * over[m] + 0.55 * np.array(c)
            img = Image.fromarray(over.astype(np.uint8))
            draw = ImageDraw.Draw(img)
            for y in range(4, role.shape[0], 10):
                for x in range(4, role.shape[1], 10):
                    if role[y, x] in (1, 2, 3):
                        du, dv = uvd[y, x, :2]
                        draw.line([(x, y), (x + du, y + dv)], fill=(255, 255, 0), width=1)
            rows.append(np.concatenate([np.asarray(img), nxt], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print(out)


if __name__ == "__main__":
    ep = Path(sys.argv[1])
    panel(ep, Path(sys.argv[2]) if len(sys.argv) > 2 else ep / "qc_gt4d.png")
