#!/usr/bin/env python
"""Which way do the Kling camera extrinsics go?  Compare measured background flow with the flow a pure
head rotation predicts under each convention (world->cam vs cam->world).

For a static point at pixel p in frame a, the rotation-only prediction of its pixel in frame b is
p_b = K R^T K^-1 p (R with X_a = R X_b + t).  The wrong convention predicts the opposite direction.
Background = pixels outside the WiLoR hand boxes (dilated).  Reports per-clip cosine agreement of both
hypotheses on clips whose rotation over 4 frames exceeds a threshold, and the pooled verdict.

    /usr/bin/python3.10 scripts/human_pretrain/check_kling_camera.py --clips 40 --min-deg 0.8
"""
from __future__ import annotations

import argparse
import io
import json

import cv2
import numpy as np
import pyarrow.parquet as pq

from metiswam4d.data.human.hand_render import relative_pose
from metiswam4d.data.human.kling import KLING_ROOT, decode_frames, load_camera_npz, read_bytes
from metiswam4d.data.human.window import source_frame_map

STRIDE = 4


def rotation_flow(K: np.ndarray, R: np.ndarray, h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    p = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3).astype(np.float64)
    d = (np.linalg.inv(K) @ p.T)
    q = K @ (R.T @ d)
    q = (q[:2] / q[2:3]).T.reshape(h, w, 2)
    return q - np.stack([xs, ys], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{KLING_ROOT}/KlingHumanEgo20M_30fps_hi_handtrack_subset.parquet")
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--min-deg", type=float, default=0.8, help="minimum rotation over 4 frames to use a pair")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cols = ["blobstore_key", "video_path", "offset", "size", "width", "height", "nb_frames", "src_fps", "src_nb_frames",
            "camera_npz_archive", "camera_npz_member", "wilor_npz_archive", "wilor_npz_member"]
    t = pq.read_table(a.parquet, columns=cols)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(t.num_rows)
    agree = {"w2c": [], "c2w": []}
    used = 0
    for i in order:
        if used >= a.clips:
            break
        row = {c: t[c][int(i)].as_py() for c in cols}
        if (row["width"], row["height"]) != (512, 288):
            continue
        try:
            cam = load_camera_npz(row["camera_npz_archive"], row["camera_npz_member"])
        except Exception:  # noqa: BLE001
            continue
        E = cam["extrinsics"].astype(np.float64)
        K = cam["intrinsics"][0].astype(np.float64)
        n = int(row["nb_frames"])
        smap = source_frame_map(n, row["src_fps"], int(row["src_nb_frames"]))
        # pick the 30 fps pair with the largest rotation
        best, best_ang = None, 0.0
        for k in range(0, n - STRIDE, 2):
            fa, fb = min(int(smap[k]), len(E) - 1), min(int(smap[k + STRIDE]), len(E) - 1)
            R, _ = relative_pose(E, fa, fb, True)
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if ang > best_ang:
                best, best_ang = (k, fa, fb), ang
        if best is None or best_ang < a.min_deg:
            continue
        k, fa, fb = best
        try:
            fr = decode_frames(read_bytes(row["video_path"], row["offset"], row["size"]), [k, k + STRIDE])
        except Exception:  # noqa: BLE001
            continue
        g0, g1 = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 3, 5, 1.2, 0)
        # background: outside dilated hand boxes at the source frame
        mask = np.ones((288, 512), dtype=bool)
        try:
            with open(row["wilor_npz_archive"], "rb"):
                pass
            import tarfile
            with tarfile.open(row["wilor_npz_archive"]) as tf:
                z = np.load(io.BytesIO(tf.extractfile(row["wilor_npz_member"]).read()))
            for f_idx, box in zip(z["frame_index"], z["box"]):
                if int(f_idx) in (fa, fb):
                    x0, y0, x1, y1 = (int(v) for v in box)
                    mask[max(y0 - 20, 0):y1 + 20, max(x0 - 20, 0):x1 + 20] = False
        except Exception:  # noqa: BLE001
            pass
        mag = np.linalg.norm(flow, axis=-1)
        mask &= (mag > 0.3) & (mag < 30)
        if mask.sum() < 2000:
            continue
        f_meas = flow[mask]
        for name, w2c in (("w2c", True), ("c2w", False)):
            R, _ = relative_pose(E, fa, fb, w2c)
            f_pred = rotation_flow(K, R, 288, 512)[mask]
            cos = (f_meas * f_pred).sum(-1) / (np.linalg.norm(f_meas, axis=-1) * np.linalg.norm(f_pred, axis=-1) + 1e-6)
            agree[name].append(float(np.median(cos)))
        used += 1
        print(f"{row['blobstore_key'].split(':')[-1][:40]:42s} pair {fa}->{fb} rot {best_ang:.2f} deg  "
              f"bg px {int(mask.sum()):6d}  cos w2c {agree['w2c'][-1]:+.2f}  c2w {agree['c2w'][-1]:+.2f}")
    summary = {k: {"median_cos": float(np.median(v)) if v else None, "n": len(v)} for k, v in agree.items()}
    verdict = max(summary, key=lambda k: summary[k]["median_cos"] or -2)
    print(json.dumps({"summary": summary, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
