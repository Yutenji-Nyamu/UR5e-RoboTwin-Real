#!/usr/bin/env python
"""Is the translation in the Kling camera_npz metric?  Compare it against DA3's own metric pose estimate.

For each sampled clip, 8 frames at stride 4 go through DA3 any-view inference (metric depth + per-view
extrinsics).  For every consecutive pair we compare the relative camera translation from the Kling npz
with DA3's: magnitude ratio |t_kling| / |t_da3| and direction cosine (both w2c, X_a = R X_b + t).
A ratio pinned near a constant across clips means the Kling translation is metric up to that constant;
a wide spread means it cannot be used as a supervised target.

    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.10 scripts/human_pretrain/check_kling_translation.py --clips 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import ig10k_human_depth_masks as DM  # noqa: E402  (da3_model)

from metiswam4d.data.human.hand_render import relative_pose  # noqa: E402
from metiswam4d.data.human.kling import KLING_ROOT, decode_frames, load_camera_npz, read_bytes  # noqa: E402
from metiswam4d.data.human.window import source_frame_map  # noqa: E402

STRIDE = 4
VIEWS = 8
MIN_T = 0.005  # m: ignore pairs where DA3 sees (almost) no translation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{KLING_ROOT}/KlingHumanEgo20M_30fps_hi_handtrack_subset.parquet")
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cols = ["blobstore_key", "video_path", "offset", "size", "width", "height", "nb_frames", "src_fps", "src_nb_frames",
            "camera_npz_archive", "camera_npz_member"]
    t = pq.read_table(a.parquet, columns=cols)
    rng = np.random.default_rng(a.seed)
    model, torch = DM.da3_model()
    ratios, cosines, per_clip = [], [], []
    used = 0
    for i in rng.permutation(t.num_rows):
        if used >= a.clips:
            break
        row = {c: t[c][int(i)].as_py() for c in cols}
        n = int(row["nb_frames"])
        if (row["width"], row["height"]) != (512, 288) or n < STRIDE * VIEWS + 1:
            continue
        try:
            cam = load_camera_npz(row["camera_npz_archive"], row["camera_npz_member"])
            E = cam["extrinsics"].astype(np.float64)
            smap = source_frame_map(n, row["src_fps"], int(row["src_nb_frames"]))
            k0 = int(rng.integers(0, n - STRIDE * VIEWS))
            ks = [k0 + STRIDE * j for j in range(VIEWS)]
            fr = decode_frames(read_bytes(row["video_path"], row["offset"], row["size"]), ks)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {row['blobstore_key'][-30:]}: {exc!r}")
            continue
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.inference(list(fr), process_res=DM.DA3_PROCESS_RES, process_res_method=DM.DA3_RES_METHOD)
        Ed = np.asarray(pred.extrinsics, dtype=np.float64)[:, :3, :]
        r_clip, c_clip = [], []
        for j in range(VIEWS - 1):
            fa, fb = min(int(smap[ks[j]]), len(E) - 1), min(int(smap[ks[j + 1]]), len(E) - 1)
            _, tk = relative_pose(E, fa, fb, True)
            _, td = relative_pose(Ed, j, j + 1, True)
            nd, nk = float(np.linalg.norm(td)), float(np.linalg.norm(tk))
            if nd < MIN_T:
                continue
            r_clip.append(nk / nd)
            c_clip.append(float(np.dot(tk, td) / (nk * nd + 1e-9)))
        if not r_clip:
            continue
        used += 1
        ratios += r_clip
        cosines += c_clip
        per_clip.append({"clip": row["blobstore_key"][-40:], "pairs": len(r_clip), "ratio_median": float(np.median(r_clip)),
                         "cos_median": float(np.median(c_clip)), "da3_metric": int(pred.is_metric),
                         "kling_t_mm_median": float(np.median([np.linalg.norm(relative_pose(E, min(int(smap[ks[j]]), len(E) - 1),
                                                                                              min(int(smap[ks[j + 1]]), len(E) - 1), True)[1])
                                                                for j in range(VIEWS - 1)]) * 1000)})
        p = per_clip[-1]
        print(f"{p['clip']:42s} pairs {p['pairs']}  |t_kling| {p['kling_t_mm_median']:6.1f} mm  ratio {p['ratio_median']:7.3f}  "
              f"cos {p['cos_median']:+.2f}", flush=True)
    r = np.asarray(ratios)
    summary = {"clips": used, "pairs": len(ratios),
               "ratio_median": float(np.median(r)), "ratio_q25": float(np.percentile(r, 25)), "ratio_q75": float(np.percentile(r, 75)),
               "log10_ratio_std": float(np.std(np.log10(r + 1e-9))),
               "cos_median": float(np.median(cosines)), "cos_frac_gt_0.5": float(np.mean(np.asarray(cosines) > 0.5))}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
