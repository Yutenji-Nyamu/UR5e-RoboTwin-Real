#!/usr/bin/env python
"""Render IG-10K Track4D (objects + hands) into the frozen mu31/yuv444p RGB video the trainer reads.

training/raw_dataset.py validates every track video against a frozen contract:
    schema  janustrack4d.rgb_video.v3      codec_id janustrack4d.signed_mu_law_xyz_rgb.mu31.yuv444p.v1
    libx264 / yuv444p / mu = 31 / scale_xyz_metres = (0.022, 0.0135, 0.0165)
so IG-10K uses exactly those constants (dataset-specific p99 scales would be rejected).  Frame t of
the video encodes delta(t -> t+4) for source frame t; the four stride-4 phases merged cover every
t in [0, n-4).  Invalid -> (0,0,0); zero motion -> (128,128,128).

Per task dir:
    track_rgb/ep_XXXXX.mp4 + ep_XXXXX.codec.json      one per episode (what training loads)
    review.mp4                                        first REVIEW_EPISODES episodes, RGB | track
    review_XXXX.jpg                                   one frame per reviewed episode
CPU only.  Run: /usr/bin/python3.10 ig10k_render_track_rgb.py run --workers 32
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_human_depth_masks as DM  # noqa: E402
from ig10k_anno_reader import IG10KAnno, SUBSETS  # noqa: E402

SCALES = np.asarray((0.022, 0.0135, 0.0165), np.float32)  # frozen, see training/raw_dataset.py
MU = 31.0
SCHEMA = "janustrack4d.rgb_video.v3"
CODEC_ID = "janustrack4d.signed_mu_law_xyz_rgb.mu31.yuv444p.v1"
FPS = 30
STRIDE = 4
REVIEW_EPISODES = 3


def encode(delta: np.ndarray) -> np.ndarray:
    q = np.clip(delta / SCALES, -1.0, 1.0)
    e = np.sign(q) * np.log1p(MU * np.abs(q)) / np.log1p(MU)
    return np.rint((e + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def write_mp4(frames: np.ndarray, out: Path, pix_fmt="yuv444p", crf=10):
    n, h, w = frames.shape[:3]
    tmp = out.with_name(out.stem + ".tmp.mp4")
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:", "-threads", "1",
         "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(crf), "-preset", "medium", "-threads", "1",
         "-movflags", "+faststart", str(tmp)], input=frames.tobytes(), capture_output=True, timeout=1800)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr.decode()[:200]}")
    os.replace(tmp, out)


def render_episode(args):
    sub, task, ep, out_dir = args
    try:
        a = IG10KAnno()
        tr = a.load_track4d(sub, task, ep)
        src, delta, valid = tr["src"], tr["delta"], tr["valid"]
        n_pairs, h, w = delta.shape[:3]
        assert np.array_equal(src, np.arange(n_pairs)), "expected contiguous source frames 0..n-5"
        rgb = encode(delta)
        rgb[~valid] = 0
        out = Path(out_dir) / f"ep_{ep:05d}.mp4"
        write_mp4(rgb, out)
        meta = {
            "schema": SCHEMA, "codec_id": CODEC_ID, "codec": "libx264", "pixel_format": "yuv444p",
            "crf": 10, "mu": MU, "fps": float(FPS), "frames": int(n_pairs),
            "scale_xyz_metres": [float(x) for x in SCALES],
            "normalization_scope": "frozen trainer constant TRACK_VIDEO_SCALES (shared with robot data)",
            "rgb_channels": ["delta_x_cam", "delta_y_cam", "delta_z_cam"],
            "invalid_rgb": [0, 0, 0], "zero_motion_rgb": [128, 128, 128],
            "frame_alignment": f"video frame t = delta(source t -> t+{STRIDE}); source frames 0..n-{STRIDE + 1}; "
                               f"no padded tail frames",
            "lossless_reference": "track4d.h5 (+ hand_track4d.h5) via ig10k_anno_reader.load_track4d",
            "validity": "track4d.h5::valid_bits | hand_track4d.h5::valid_bits",
            "content": "manipulated objects (RAFT+DA3) and, where MANO exists, hand surface (MANO mesh)",
            "hand_pixels_present": bool(tr["hand"] is not None and tr["hand"].any()),
        }
        out.with_suffix(".codec.json").write_text(json.dumps(meta, indent=2))
        return (ep, int(n_pairs), int(valid.sum()), "")
    except Exception as e:  # noqa: BLE001
        return (ep, 0, 0, f"{type(e).__name__}: {e}"[:300])


def review(sub: str, task: str, eps: list[int], a: IG10KAnno, td: Path):
    """RGB | track side by side for the first few episodes, plus one jpg per episode."""
    import cv2
    info = json.loads((a.video_root / sub / task / "meta" / "info.json").read_text())
    ego = DM.ego_key(info)
    rows = {r["episode"]: r for r in DM.episodes_of(a.video_root / sub / task, ego)}
    frames_all = []
    for ep in eps:
        clip = Path("/dev/shm") / f"review_{os.getpid()}_{ep}.mp4"
        try:
            DM.cut_episode(rows[ep], clip)
            rgb = DM.read_frames(clip)
        finally:
            clip.unlink(missing_ok=True)
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(td / "track_rgb" / f"ep_{ep:05d}.mp4"),
                            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True)
        trk = np.frombuffer(p.stdout, np.uint8).reshape(-1, rgb.shape[1], rgb.shape[2], 3)
        n = min(len(rgb), len(trk))
        side = np.concatenate([rgb[:n], trk[:n]], 2)
        frames_all.append(side)
        k = int(np.argmax((trk[:n] > 0).any(-1).reshape(n, -1).mean(1)))
        cv2.imwrite(str(td / f"review_{ep:04d}.jpg"), cv2.cvtColor(side[k], cv2.COLOR_RGB2BGR))
    write_mp4(np.concatenate(frames_all), td / "review.mp4", pix_fmt="yuv420p", crf=18)


def cmd_run(args):
    a = IG10KAnno()
    A = a.anno_root
    (A / "_logs").mkdir(exist_ok=True)
    logfh = open(A / "_logs" / "render_track_rgb.log", "a")
    dirs = [(sub, p.name) for sub in SUBSETS for p in sorted((A / sub).glob("*")) if (p / "track4d.h5").exists()]
    if args.only:
        dirs = [d for d in dirs if d[1] in args.only.split(",")]
    pool = mp.get_context("fork").Pool(args.workers)
    try:
        for sub, task in dirs:
            td = A / sub / task
            out_dir = td / "track_rgb"
            eps = a.episode_ids(sub, task)
            todo = [e for e in eps if not (out_dir / f"ep_{e:05d}.codec.json").exists()]
            if not todo and (td / "review.mp4").exists():
                continue
            out_dir.mkdir(exist_ok=True)
            t0 = time.time()
            res = pool.map(render_episode, [(sub, task, e, str(out_dir)) for e in todo], chunksize=1) if todo else []
            fails = [r for r in res if r[3]]
            for r in fails:
                DM.log(f"  FAIL {sub}/{task} ep{r[0]}: {r[3]}", logfh)
            try:
                review(sub, task, eps[:REVIEW_EPISODES], a, td)
            except Exception:  # noqa: BLE001
                DM.log(f"  review failed {sub}/{task}:\n{traceback.format_exc()}", logfh)
            mb = sum(f.stat().st_size for f in out_dir.glob("*.mp4")) / 1e6
            DM.log(f"DONE {sub}/{task}: {len(eps)} eps ({len(todo)} rendered, {len(fails)} failed) "
                   f"{(time.time() - t0) / 60:.1f} min, {mb:.0f} MB", logfh)
    finally:
        pool.close()
        pool.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--only", default="", help="comma list of task dir names (demo)")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
