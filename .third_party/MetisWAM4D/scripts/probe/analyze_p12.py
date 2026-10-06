#!/usr/bin/env python
"""P1.2 - region-wise quality of the predicted future RGB (head view) of the pixel-centric baselines.

For every P3.1 window with a decoded prediction (``pred_rgb``), the head-camera part of the predicted frames
is compared with the GT head frames at ``start + 4k`` (k = 1..8), split by the GT masks into body / object /
manipulated-object / interaction-zone / background regions: PSNR and mean absolute error per region, and the
"object-motion recall": fraction of the GT moving-object pixels whose predicted frame changed vs. the current
frame (does the prediction move the object at all?).

Run with $PY (needs the vendored openwam layout helper only for the T-layout geometry; here we just slice).

    source scripts/probe/env.sh; python3 scripts/probe/analyze_p12.py --model alpha
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
BATCH = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
GT = OUT / "gt_events"
H, W = 240, 320

# where the head view sits in each model's decoded frame and its size
HEAD_CROP = {
    "alpha": lambda img: cv2.resize(img[:256, :320], (W, H), interpolation=cv2.INTER_AREA),     # T-layout 384x320
    "effwam": lambda img: img[:240, :320],                                                       # head native 240x320 on top
    "flowwam": lambda img: cv2.resize(img[:256, :320], (W, H), interpolation=cv2.INTER_AREA),   # T-tile 384x320
    "xwam": lambda img: cv2.resize(img[:, :320], (W, H), interpolation=cv2.INTER_AREA),         # [256, 960]: head | left | right
}


def psnr(a, b, m):
    if m.sum() < 10:
        return np.nan
    mse = ((a[m].astype(np.float32) - b[m].astype(np.float32)) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alpha")
    ap.add_argument("--root", default=str(OUT / "p31"))
    a = ap.parse_args()
    d = Path(a.root) / a.model
    crop = HEAD_CROP[a.model]
    acc = defaultdict(list)
    per_window = []
    files = sorted(d.glob("*.npz"))
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "pred_rgb" not in z.files or z["pred_rgb"].size == 0:
            continue
        meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
        task, variant, name = meta["key"].split("/")
        start = meta["start"]
        ep = BATCH / task / variant / name
        pred = z["pred_rgb"]
        n_pred = pred.shape[0]
        # frame index k of the prediction corresponds to raw frame start + 4k (k=0 is the conditioning frame)
        ks = [k for k in range(1, n_pred) if True]
        with h5py.File(ep / "source.hdf5", "r") as src, h5py.File(ep / "masks.h5", "r") as mk:
            T = src["observation/head_camera/depth"].shape[0]
            frames = [min(start + 4 * k, T - 1) for k in ks]
            gt = [cv2.cvtColor(cv2.imdecode(np.frombuffer(src["observation/head_camera/rgb"][fr], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for fr in frames]
            cur = cv2.cvtColor(cv2.imdecode(np.frombuffer(src["observation/head_camera/rgb"][start], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            bits = np.unpackbits(mk["head_camera/mask_bits"][frames], axis=-1, bitorder="little")[..., :W].astype(bool)
        gz = np.load(GT / task / variant / f"{name}.npz", allow_pickle=False)
        inter = np.unpackbits(gz["interaction_pix"][frames], axis=-1, bitorder="little")[..., :W].astype(bool)
        moving = np.unpackbits(gz["moving_pix"][frames], axis=-1, bitorder="little")[..., :W].astype(bool)
        row = dict(key=meta["key"], start=start, kind=meta["kind"])
        vals = defaultdict(list)
        for i, k in enumerate(ks):
            p = crop(pred[k])
            g = gt[i]
            body, obj = bits[i, 0], bits[i, 1] & ~bits[i, 0]
            bg = ~(bits[i, 0] | bits[i, 1])
            for nm, m in (("body", body), ("object", obj), ("background", bg), ("interaction", inter[i]), ("moving_object", moving[i])):
                vals[f"psnr_{nm}"].append(psnr(p, g, m))
                vals[f"mae_{nm}"].append(float(np.abs(p[m].astype(np.float32) - g[m].astype(np.float32)).mean()) if m.sum() >= 10 else np.nan)
            # copy-the-present baseline: how much better than just repeating the current frame?
            for nm, m in (("body", body), ("object", obj), ("moving_object", moving[i])):
                vals[f"psnr_copy_{nm}"].append(psnr(cur, g, m))
            # object-motion recall: moving GT pixels where prediction differs from the current frame by > 20/255
            if moving[i].sum() >= 10:
                changed = np.abs(p.astype(np.float32) - cur.astype(np.float32)).mean(-1) > 20
                vals["motion_recall"].append(float(changed[moving[i]].mean()))
                static_ref = ~(bits[i, 0] | bits[i, 1] | moving[i])
                vals["motion_false_alarm"].append(float(changed[static_ref].mean()))
        for k2, v in vals.items():
            v = np.asarray(v, dtype=np.float64)
            if np.isfinite(v).any():
                row[k2] = float(np.nanmean(v))
                acc[k2].append(row[k2])
                acc[f"{k2}__{meta['kind']}"].append(row[k2])
        per_window.append(row)
    summary = {k: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v))) for k, v in acc.items()}
    out_dir = d / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "p12_regions.json").write_text(json.dumps(summary, indent=1))
    print(f"[{a.model}] {len(per_window)} windows with decoded predictions")
    for k in ("psnr_body", "psnr_object", "psnr_moving_object", "psnr_interaction", "psnr_background",
              "psnr_copy_body", "psnr_copy_object", "psnr_copy_moving_object", "motion_recall", "motion_false_alarm"):
        if k in summary:
            print(f"  {k:24s} mean {summary[k]['mean']:6.2f}  median {summary[k]['median']:6.2f}  (n={summary[k]['n']})")


if __name__ == "__main__":
    main()
