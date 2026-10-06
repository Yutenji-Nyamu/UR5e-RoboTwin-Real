#!/usr/bin/env python3
"""Per-head focus: is there ANY attention head that specialises on the hand-object interaction or the object?

Uses the head-resolved attention saved at three denoising steps (``attn_by_head`` [3, L, h, K]).  For every
(step, layer, head) the lift of the interaction zone / manipulated object / arm is computed per window; we report
the distribution over heads of the window-median lift, the best head, and how many heads exceed lift 2 / 3.
If even the most interaction-focused head is weak, averaging is not hiding a specialised reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_openloop import window_gt  # noqa: E402

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alpha")
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    d = OUT / "p31" / a.model
    files = sorted(d.glob("*.npz"))[: a.limit]
    lifts = {"inter": [], "manip": [], "body": [], "obj": []}
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
        seg = json.loads(str(z["segments"]))
        r = seg["ranges"]
        fv, vh, vw = seg["video_grid"]
        try:
            gt = window_gt(meta["key"], meta["start"])
        except Exception:  # noqa: BLE001
            continue
        A = z["attn_by_head"].astype(np.float32)                       # [S, L, h, K]
        vf = A[..., r["video_future"][0]:r["video_future"][1]].reshape(A.shape[0], A.shape[1], A.shape[2], fv - 1, vh, vw)[..., :8, :]
        vf = vf / np.maximum(vf.sum(axis=(3, 4, 5), keepdims=True), 1e-12)   # per (step, layer, head) distribution over head-view future tokens
        masks = {"inter": gt["video"]["inter"] > 0, "manip": gt["video"]["moving"] > 0.05, "body": gt["video"]["body"] > 0.2,
                 "obj": (gt["video"]["obj"] > 0.2)}
        for nm, m in masks.items():
            share = m.mean()
            if share == 0:
                continue
            mass = (vf * m[None, None, None]).sum(axis=(3, 4, 5))         # [S, L, h]
            lifts[nm].append(mass / share)
    out = {}
    print(f"[{a.model}] per-head lift over {len(files)} windows (median over windows; heads = steps x layers x heads)")
    for nm, arr in lifts.items():
        if not arr:
            continue
        L = np.stack(arr)                                                 # [N, S, L, h]
        med = np.median(L, axis=0)                                        # [S, L, h]
        flat = med.reshape(-1)
        best = np.unravel_index(np.argmax(med), med.shape)
        out[nm] = dict(heads=int(flat.size), median_of_heads=float(np.median(flat)), p90=float(np.percentile(flat, 90)), max=float(flat.max()),
                       best_step_layer_head=[int(x) for x in best], frac_heads_gt2=float((flat > 2).mean()), frac_heads_gt3=float((flat > 3).mean()),
                       mean_over_heads_of_window_mean=float(np.mean(L)))
        print(f"  {nm:6s} median head {out[nm]['median_of_heads']:.2f}  p90 {out[nm]['p90']:.2f}  best {out[nm]['max']:.2f} at (step,layer,head)={best}"
              f"  heads>2: {out[nm]['frac_heads_gt2']:.1%}  heads>3: {out[nm]['frac_heads_gt3']:.1%}")
    (d / "analysis" / "heads.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
