"""Export a few RT2_MetisWAM4D episodes as videos for eyeballing.

Per episode (all source frames, 30 fps):
  <out>/<task>__<variant>__epN/
    video_3view.mp4     three-view L layout 384x320 exactly as the Video expert sees it
    track4d.mp4         head RGB | Track4D mu-law RGB (t -> t+4, training encoding) | role overlay
    phases.png          the 9 stride-4 Track4D frames of one training window (anchor + 8 transitions)
    info.json           displacement statistics of the episode

  python scripts/data_prep/rt2/export_episode_videos.py --out /path --episodes adjust_bottle/demo_clean_4d/12 ...
  python scripts/data_prep/rt2/export_episode_videos.py --out /path --sample 6 --seed 0   # random tasks, clean + randomized
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from metiswam4d.data.rt2.codec import CODEC_ID, SHRINK_D_M, anchor_frame, encode_uvd, uvd_from_xyz  # noqa: E402
from metiswam4d.data.rt2.episode_dataset import STRIDE, decode_jpeg, multiview_layout  # noqa: E402

DATA = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D")
CAMS = ("head_camera", "left_camera", "right_camera")
ROLE_COLOR = np.array([[0, 0, 0], [60, 140, 255], [255, 120, 40]], np.uint8)  # bg, robot, object


def write_mp4(path: Path, frames: np.ndarray, fps: int = 30) -> None:
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def export(task: str, variant: str, episode: int, out: Path) -> dict:
    d = DATA / task / variant / f"episode{episode}"
    dest = out / f"{task}__{variant}__ep{episode}"
    dest.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
        n_tr = t4d["delta_xyz_cam"].shape[0]
        n = n_tr + STRIDE
        rgb = {c: [np.asarray(decode_jpeg(src[f"observation/{c}/rgb"][t])) for t in range(n)] for c in CAMS}
        video = np.stack([multiview_layout({c: Image.fromarray(rgb[c][t]) for c in CAMS}, CAMS, 384, 320) for t in range(n)])
        delta = t4d["delta_xyz_cam"][:].astype(np.float32)          # [n_tr, H, W, 3]
        valid = t4d["valid"][:] > 0
        role = t4d["role"][:n_tr]
        depth_m = src["observation/head_camera/depth"][:n_tr].astype(np.float32) / 1000.0
        K = np.asarray(src["observation/head_camera/intrinsic_cv"][0], np.float64)
    fg = valid & (role > 0)
    track_rgb, _ = encode_uvd(uvd_from_xyz(delta, depth_m, K), fg, fg.shape[-1], SHRINK_D_M["rt2"])  # training encoding
    head = np.stack([rgb["head_camera"][t] for t in range(n_tr)])
    if head.shape[1:3] != track_rgb.shape[1:3]:
        head = np.stack([np.asarray(Image.fromarray(f).resize(track_rgb.shape[2:0:-1], Image.BILINEAR)) for f in head])
    overlay = (0.55 * head + 0.45 * ROLE_COLOR[role]).astype(np.uint8)
    panel = np.concatenate((head, track_rgb, overlay), axis=2)
    write_mp4(dest / "video_3view.mp4", video)
    write_mp4(dest / "track4d.mp4", panel)
    # One training window: anchor (zero displacement, foreground grey) + 8 stride-4 transitions.
    s = max(0, (n - 33) // 2)
    idx = [s + STRIDE * k for k in range(8)]
    phases = np.concatenate([anchor_frame(role[idx[0]] > 0)] + [track_rgb[t] for t in idx], axis=1)
    Image.fromarray(np.concatenate((np.concatenate([head[idx[0]]] + [head[t] for t in idx], axis=1), phases), axis=0)).save(dest / "phases.png")
    mag = np.linalg.norm(delta, axis=-1)
    info = {"task": task, "variant": variant, "episode": episode, "frames": int(n),
            "foreground_fraction": float(fg.mean()), "robot_fraction": float((role == 1).mean()),
            "object_fraction": float((role == 2).mean()),
            "background_nonzero_fraction": float((mag[valid & (role == 0)] > 1e-4).mean()) if (valid & (role == 0)).any() else None,
            "fg_disp_m_p50_p99_max": [float(np.percentile(mag[fg], 50)), float(np.percentile(mag[fg], 99)), float(mag[fg].max())] if fg.any() else None,
            "codec": CODEC_ID, "window_start": s}
    (dest / "info.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", nargs="*", default=[], help="task/variant/episode")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    picks = [e.split("/") for e in args.episodes]
    if args.sample:
        rows = [json.loads(l) for l in open(DATA / "index.jsonl") if l.strip()]
        rng = random.Random(args.seed)
        tasks = rng.sample(sorted({r["task"] for r in rows}), args.sample)
        for i, task in enumerate(tasks):
            variant = "demo_clean_4d" if i % 2 == 0 else "demo_randomized_4d"
            cands = [r for r in rows if r["task"] == task and r["variant"] == variant]
            r = rng.choice(cands)
            picks.append([task, variant, str(r["episode"])])
    out = Path(args.out)
    for task, variant, ep in picks:
        info = export(task, variant, int(ep), out)
        print(json.dumps(info))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
