"""Review video for generated VLABench episodes: per frame, top row = the three policy views (forward / wrist / right,
480x480 -> 240x240 each), bottom row = head RGB on the 240x320 track grid with role overlay (blue robot, red object),
the Track uvd frame t -> t+4 as the training RGB (mu-law codec, black = static / not foreground), and |du, dv| on the
robot / object pixels.

    /usr/bin/python3.10 scripts/vlabench/qc_video.py --n 6 [--root ROOT] [--out DIR]
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import random
import sys

import cv2
import h5py
import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metiswam4d.data.rt2.codec import SHRINK_D_M, encode_uvd  # noqa: E402

FPS = 10


def jpeg(raw) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw.tobytes())).convert("RGB"))


def label(img: np.ndarray, text: str, bottom: bool = False) -> np.ndarray:
    img = np.ascontiguousarray(img)
    org = (6, img.shape[0] - 8) if bottom else (6, 18)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render(path: Path, out: Path) -> np.ndarray:
    frames, rgb3_frames, track_frames = [], [], []
    with h5py.File(path) as h:
        n = int(h.attrs["frames"])
        title = f"{h.attrs['task']}/{path.stem}: {h.attrs['instruction']}"
        delta, role = h["delta_uvd"], h["role"]
        gh, gw = role.shape[1:]
        for t in range(n - 4):
            views = [cv2.resize(jpeg(h[f"{c}_rgb"][t]), (240, 240), interpolation=cv2.INTER_AREA)
                     for c in ("head", "left", "right")]
            top = np.concatenate([label(v, name) for v, name in zip(views, ("forward (head)", "wrist", "right"))], 1)
            head = cv2.resize(jpeg(h["head_rgb"][t]), (gw, gh), interpolation=cv2.INTER_AREA)
            ro = role[t]
            over = head.copy()
            over[ro == 1] = (0.45 * over[ro == 1] + [0, 0, 140]).astype(np.uint8)
            over[ro == 2] = (0.45 * over[ro == 2] + [140, 0, 0]).astype(np.uint8)
            d = delta[t].astype(np.float32)
            rgb, _ = encode_uvd(d, ro > 0, gw, SHRINK_D_M["robodojo"])
            mag = np.hypot(d[..., 0], d[..., 1])
            mag = cv2.applyColorMap(np.clip(mag / 30.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)[..., ::-1]
            mag[ro == 0] = 0
            bottom = np.concatenate([label(over, "role: robot / object"), label(rgb, "track uvd t->t+4"),
                                     label(mag, "|du,dv| (0-30 px)")], 1)
            bottom = cv2.resize(bottom, (top.shape[1], int(bottom.shape[0] * top.shape[1] / bottom.shape[1])))
            frame = np.concatenate([top, bottom], 0)
            frames.append(label(frame, f"{title}  t={t}/{n}", bottom=True))
            rgb3 = np.concatenate([jpeg(h[f"{c}_rgb"][t]) for c in ("head", "left", "right")], 1)  # 480 x 1440
            rgb3_frames.append(label(rgb3, f"forward | wrist | right    {title}  t={t}/{n}", bottom=True))
            big = lambda x: cv2.resize(x, (gw * 2, gh * 2), interpolation=cv2.INTER_NEAREST)
            track_frames.append(label(np.concatenate([big(over), big(rgb), big(mag)], 1),
                                      f"role | track uvd t->t+4 | |du,dv|    {title}  t={t}/{n}", bottom=True))
    for name, seq in (("", frames), ("_rgb3", rgb3_frames), ("_track", track_frames)):
        seq = [np.pad(f, ((0, (-f.shape[0]) % 16), (0, (-f.shape[1]) % 16), (0, 0))) for f in seq]
        imageio.mimwrite(out.with_name(out.stem + name + ".mp4"), seq, fps=FPS, codec="libx264", quality=8,
                         macro_block_size=16)
    print(f"{out}  ({title}, {len(frames)} frames)", flush=True)
    return frames[len(frames) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ytech_milm_intern/danglingwei/datas/VLABench/gen4d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out or root / "_qc" / "videos")
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "text_cache")
    rng = random.Random(args.seed)
    picks = rng.sample(tasks, min(args.n, len(tasks)))
    stills = []
    for task in picks:
        ep = rng.randrange(500)
        path = root / task / f"episode{ep}.h5"
        stills.append(render(path, out / f"{task}_episode{ep}.mp4"))
    w = max(s.shape[1] for s in stills)
    sheet = np.concatenate([np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0))) for s in stills], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / "contact_sheet.png")
    print(out / "contact_sheet.png")


if __name__ == "__main__":
    main()
