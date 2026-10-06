#!/usr/bin/env python
"""KlingHumanEgo-2.5M-5000H `hi`: hand-only Track4D video from the existing WiLoR estimates.

Dense per-pixel Track4D (DA3 + RAFT + SAM) is off the table for Kling -- 8e8 frames at the
measured 1.45 s/frame is ~190 GPU-days.  What Kling already carries on 100% of the `hi`
subset is a WiLoR pass per clip: 21 3D hand joints per detected hand per source frame, their
2D projections in the 512x288 frame, and a per-clip `track_id` that keeps hand identity across
frames.  That is enough for a hand Track4D: displacement of every joint between t and t+4, in
WiLoR's camera frame, rendered onto the hand's footprint.

Convention (same as the robot shards, JanusTrack4d/preprocess/track3d_codec.py):
    q   = clip(delta_xyz / scale_xyz, -1, 1)          scale_xyz: dataset-global, per axis
    e   = sign(q) * log1p(mu*|q|) / log1p(mu)          mu = 31
    rgb = rint((e + 1) * 127.5)                        zero motion -> (128,128,128) grey
    invalid / untracked -> (0,0,0) black
Stride-4, four phases: frame t of the track video encodes delta(t -> t+4) for every t; the
last four frames (no target) are black.

What "camera frame" means here: WiLoR's `joints_3d + camera_translation`.  The hand's own
articulation is MANO-metric; the translation lives in WiLoR's weak-perspective camera
(median Z ~6.7 m, not real depth), so absolute Z is a prior, not calibrated geometry.  This
is written into the sidecar.

Frame alignment: WiLoR `frame_index` indexes SOURCE frames; the 30 fps re-encode has
round(k * src_fps / 30) as the source frame of output frame k.

Output mirrors the video shards so we never create 3.4M small files:
    {out_root}/track_hi/job-XXXXXXXXX/shard-XXXXXX.tar     members {blobstore_key}.track.mp4
    {out_root}/track_hi/index/job-XXXXXXXXX/shard-XXXXXX.parquet   key -> (path, offset, size, ...)
    {out_root}/track_hi/codec.json                         scale_xyz_metres, mu, conventions

Sub-commands: scale (estimate scale_xyz from a sample), run, status, merge.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import tarfile
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC_PARQUET = "/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M/KlingHumanEgo20M_30fps_hi.parquet"
OUT_ROOT = "/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M"
TAG = "track_hi"
STRIDE = 4
MU = 31.0
OUT_FPS = 30
MIN_CONF = 0.3
# OpenPose 21-joint hand: wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
PALM = [0, 1, 5, 9, 13, 17]


# --------------------------------------------------------------------------- codec (parity with track3d_codec.py)
def encode(delta_xyz: np.ndarray, scale: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(delta_xyz, np.float32) / scale, -1.0, 1.0)
    e = np.sign(q) * np.log1p(MU * np.abs(q)) / np.log1p(MU)
    return np.rint((e + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- data access
_TARS = {}


def npz_from_tar(archive: str, member: str):
    tf = _TARS.get(archive)
    if tf is None:
        if len(_TARS) > 8:
            for t in _TARS.values():
                t.close()
            _TARS.clear()
        tf = _TARS[archive] = tarfile.open(archive)
    return np.load(io.BytesIO(tf.extractfile(member).read()))


def hands_by_source_frame(z) -> dict:
    """{source_frame: {track_id: (joints_cam (21,3), joints_2d (21,2))}} for confident hands."""
    out = {}
    fi = z["frame_index"]
    conf = z["confidence"]
    ok = conf >= MIN_CONF
    # Root translation in Z is WiLoR's weak-perspective pseudo-depth (p99 of |dZ| over 4 frames
    # measured at 2.65 m vs 8 cm for X/Y): pure bbox-size jitter, not motion.  Zero it, so the
    # Z channel carries only the (MANO-metric) intra-hand articulation; X/Y translation stays.
    ct = z["camera_translation"].copy()
    ct[:, 2] = 0.0
    j3 = z["joints_3d"] + ct[:, None, :]
    j2 = z["joints_2d"]
    # older WiLoR dumps have no tracker; fall back to handedness as the identity (one left, one
    # right per frame), keeping the more confident detection on a collision
    tid = z["track_id"] if "track_id" in z.files else z["is_right"].astype(np.int32)
    best = {}
    for i in np.nonzero(ok)[0]:
        key = (int(fi[i]), int(tid[i]))
        if key not in best or conf[i] > conf[best[key]]:
            best[key] = i
    for (f, t), i in best.items():
        out.setdefault(f, {})[t] = (j3[i], j2[i])
    return out


def src_frame_map(n_out: int, src_fps: str, src_n: int) -> np.ndarray:
    f = float(Fraction(src_fps))
    return np.minimum(np.rint(np.arange(n_out) * f / OUT_FPS).astype(int), max(src_n - 1, 0))


# --------------------------------------------------------------------------- rendering
def draw_hand(img: np.ndarray, pts: np.ndarray, delta: np.ndarray, scale: np.ndarray):
    """Paint one hand: palm polygon + thick finger bones + joint discs, coloured by displacement."""
    import cv2

    p = np.rint(pts).astype(np.int32)
    ext = float(np.ptp(pts, axis=0).max())
    thick = int(max(2, round(0.09 * ext)))
    col = encode(delta, scale)  # (21,3)
    palm_col = tuple(int(c) for c in encode(delta[PALM].mean(0), scale))
    cv2.fillConvexPoly(img, cv2.convexHull(p[PALM]), palm_col, lineType=cv2.LINE_8)
    for a, b in BONES:
        c = tuple(int(v) for v in encode((delta[a] + delta[b]) * 0.5, scale))
        cv2.line(img, tuple(p[a]), tuple(p[b]), c, thick, lineType=cv2.LINE_8)
    r = max(1, thick // 2 + 1)
    for j in range(21):
        cv2.circle(img, tuple(p[j]), r, tuple(int(v) for v in col[j]), -1, lineType=cv2.LINE_8)


def render_clip(row: dict, scale: np.ndarray) -> tuple[bytes, dict]:
    """-> (mp4 bytes, stats) for one clip."""
    n, h, w = int(row["nb_frames"]), int(row["height"]), int(row["width"])
    z = npz_from_tar(row["wilor_npz_archive"], row["wilor_npz_member"])
    hands = hands_by_source_frame(z)
    smap = src_frame_map(n, row["src_fps"], int(row["src_nb_frames"]))
    frames = np.zeros((n, h, w, 3), np.uint8)  # black = invalid
    n_valid = 0
    hand_px = 0
    for k in range(n - STRIDE):
        a, b = hands.get(int(smap[k])), hands.get(int(smap[k + STRIDE]))
        if not a or not b:
            continue
        for tid, (j3a, j2a) in a.items():
            if tid not in b:
                continue
            j3b = b[tid][0]
            draw_hand(frames[k], j2a, j3b - j3a, scale)
        n_valid += 1
    hand_px = int((frames.reshape(n, -1, 3).any(-1)).sum())
    # +faststart needs a seekable output, so encode to tmpfs and read back (a normal mp4, not fragmented)
    tmp = Path("/dev/shm") / f"kling_track_{os.getpid()}.mp4"
    try:
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(OUT_FPS), "-i", "pipe:",
             "-threads", "1", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "10", "-preset", "veryfast", "-threads", "1",
             "-movflags", "+faststart", str(tmp)],
            input=frames.tobytes(), capture_output=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr.decode()[:200]}")
        data = tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)
    return data, {"frames": n, "frames_with_hand_pairs": n_valid,
                      "hand_px_per_frame": hand_px / max(n, 1),
                      "hands_detected": int(len(z["frame_index"]))}


# --------------------------------------------------------------------------- scale estimation
def cmd_scale(a):
    import pyarrow.parquet as pq

    t = pq.read_table(a.parquet, columns=["wilor_npz_archive", "wilor_npz_member", "src_fps",
                                          "src_nb_frames", "nb_frames"])
    rng = np.random.default_rng(0)
    idx = rng.choice(t.num_rows, size=min(a.sample, t.num_rows), replace=False)
    rows = t.take(idx).to_pylist()
    deltas = []
    for r in rows:
        try:
            z = npz_from_tar(r["wilor_npz_archive"], r["wilor_npz_member"])
        except Exception:  # noqa: BLE001
            continue
        hands = hands_by_source_frame(z)
        smap = src_frame_map(int(r["nb_frames"]), r["src_fps"], int(r["src_nb_frames"]))
        for k in range(int(r["nb_frames"]) - STRIDE):
            a_, b_ = hands.get(int(smap[k])), hands.get(int(smap[k + STRIDE]))
            if not a_ or not b_:
                continue
            for tid, (j3a, _) in a_.items():
                if tid in b_:
                    deltas.append(b_[tid][0] - j3a)
    d = np.abs(np.concatenate(deltas)).reshape(-1, 3)
    q = np.quantile(d, a.quantile, axis=0)
    out = Path(a.out_root) / TAG
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "scale_xyz_metres": [float(x) for x in q], "mu": MU, "fps": OUT_FPS, "stride": STRIDE,
        "codec": "libx264", "pix_fmt": "yuv444p", "crf": 10,
        "rgb_channels": ["delta_x_cam", "delta_y_cam", "delta_z_cam"],
        "invalid_rgb": [0, 0, 0], "zero_motion_rgb": [128, 128, 128],
        "forward": "q=clip(delta/scale_xyz,-1,1); e=sign(q)*log1p(mu*abs(q))/log1p(mu); rgb=(e+1)/2",
        "normalization_scope": f"dataset-global independent symmetric XYZ scales, p{a.quantile * 100:.0f} "
                               f"of |delta| over {len(rows)} sampled clips / {len(d)} joint pairs",
        "frame_of_reference": "WiLoR hand camera, axes aligned with the ego camera. delta = "
                              "(joints_3d + [tx, ty, 0])[t+4] - same[t]: intra-hand articulation is "
                              "MANO-metric in XYZ; hand-root translation is kept in X/Y only. Root Z "
                              "translation is WiLoR weak-perspective pseudo-depth (p99 |dZ| 2.65 m vs "
                              "8 cm X/Y = bbox jitter) and is zeroed, so the Z channel encodes finger/"
                              "palm depth articulation only. Ego camera motion is NOT removed.",
        "footprint": "hand only (palm hull + 21 joints + 20 bones); everything else is invalid/black",
        "source": "WiLoR npz shipped with the source parquet (100% coverage on the hi subset)",
        "frame_alignment": "track frame k <- source frame round(k*src_fps/30); frame k encodes delta(k->k+4)",
    }
    (out / "codec.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("scale_xyz_metres", "normalization_scope")}, indent=1))


# --------------------------------------------------------------------------- run
_W = {}


def _init(scale):
    _W["scale"] = np.asarray(scale, np.float32)


def _work(row):
    try:
        data, st = render_clip(row, _W["scale"])
        return {"blobstore_key": row["blobstore_key"], "ok": True, "data": data, **st, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"blobstore_key": row["blobstore_key"], "ok": False, "data": b"", "frames": 0,
                "frames_with_hand_pairs": 0, "hand_px_per_frame": 0.0, "hands_detected": 0,
                "error": f"{type(e).__name__}: {e}"[:300]}


def shard_rel(video_path: str) -> str:
    p = Path(video_path)
    return f"{p.parent.name}/{p.name}"


def cmd_run(a):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_root = Path(a.out_root) / TAG
    codec = json.loads((out_root / "codec.json").read_text())
    scale = codec["scale_xyz_metres"]
    for d in ("index", "locks", "logs"):
        (out_root / d).mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    logfh = open(out_root / "logs" / f"{host}.log", "a")

    def log(m):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        logfh.write(line + "\n")
        logfh.flush()

    cols = ["blobstore_key", "video_path", "nb_frames", "width", "height", "src_fps", "src_nb_frames",
            "wilor_npz_archive", "wilor_npz_member"]
    t = pq.read_table(a.parquet, columns=cols)
    vp = t["video_path"].combine_chunks().dictionary_encode()
    codes = vp.indices.to_numpy(zero_copy_only=False)
    shards = vp.dictionary.to_pylist()
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(shards) + 1))
    log(f"start host={host} workers={a.workers} shards={len(shards)} clips={t.num_rows} scale={scale}")

    pool = mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(scale,))
    n_done = 0
    try:
        for si in range(len(shards)):
            rel = shard_rel(shards[si])
            stem = rel[:-4]
            tar_out = out_root / f"{stem}.tar"
            idx_out = out_root / "index" / f"{stem}.parquet"
            lock = out_root / "locks" / stem.replace("/", "__")
            if idx_out.exists():
                continue
            try:
                lock.mkdir(parents=True)
            except FileExistsError:
                continue
            (lock / f"{host}.{os.getpid()}").touch()
            rows_idx = order[bounds[si]:bounds[si + 1]]
            rows = t.take(pa.array(rows_idx)).to_pylist()
            t0 = time.time()
            tar_out.parent.mkdir(parents=True, exist_ok=True)
            tmp = tar_out.with_suffix(".tar.tmp")
            recs = []
            n_fail = 0
            try:
                with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tf:
                    for r in pool.imap_unordered(_work, rows, chunksize=4):
                        data = r.pop("data")
                        if not r["ok"]:
                            n_fail += 1
                            r.update(track_path="", offset=-1, size=0)
                        else:
                            name = f"{r['blobstore_key'].split(':')[-1]}.track.mp4"
                            info = tarfile.TarInfo(name=name)
                            info.size = len(data)
                            info.mtime = 0
                            hdr = tf.offset
                            tf.addfile(info, io.BytesIO(data))
                            r.update(track_path=str(tar_out), offset=hdr + 512, size=len(data))
                        recs.append(r)
                os.replace(tmp, tar_out)
                tmp_idx = idx_out.with_suffix(".parquet.tmp")
                idx_out.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(recs), tmp_idx)
                os.replace(tmp_idx, idx_out)
                el = time.time() - t0
                log(f"DONE {rel}: {len(rows)} clips in {el / 60:.1f} min ({len(rows) / el:.1f} clip/s) "
                    f"fail={n_fail} tar={tar_out.stat().st_size / 1e6:.0f} MB")
                n_done += 1
            except Exception:  # noqa: BLE001
                log(f"ERROR {rel}:\n{traceback.format_exc()}")
                tmp.unlink(missing_ok=True)
                shutil.rmtree(lock, ignore_errors=True)
            if a.max_shards and n_done >= a.max_shards:
                break
    finally:
        pool.close()
        pool.join()
    log(f"exit host={host}: shards={n_done}")


def cmd_status(a):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    out_root = Path(a.out_root) / TAG
    idx = sorted((out_root / "index").glob("job-*/shard-*.parquet"))
    n = fail = 0
    size = 0
    for f in idx:
        t = pq.read_table(f, columns=["ok", "size"])
        n += t.num_rows
        fail += t.num_rows - pc.sum(t["ok"]).as_py()
        size += pc.sum(t["size"]).as_py() or 0
    locks = len(list((out_root / "locks").glob("*"))) if (out_root / "locks").exists() else 0
    print(f"shards {len(idx)}/5192  clips {n}  fail {fail}  out {size / 1e9:.2f} GB  in-flight {locks - len(idx)}")
    for lg in sorted((out_root / "logs").glob("*.log")):
        tail = lg.read_text().strip().splitlines()
        print(f"--- {lg.name}: {tail[-1][:120] if tail else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=SRC_PARQUET)
    ap.add_argument("--out-root", default=OUT_ROOT)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scale")
    s.add_argument("--sample", type=int, default=3000)
    s.add_argument("--quantile", type=float, default=0.99)
    s.set_defaults(fn=cmd_scale)
    r = sp.add_parser("run")
    r.add_argument("--workers", type=int, default=max(4, os.cpu_count() - 8))
    r.add_argument("--max-shards", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    st = sp.add_parser("status")
    st.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
