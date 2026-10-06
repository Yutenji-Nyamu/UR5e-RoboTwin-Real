#!/usr/bin/env python3
"""P2.3 deficiency metrics: does the baseline's Action attention focus on the hand-object interaction?

Per window (head-view future tokens, attention summed over heads/queries/steps/layers unless grouped):

  obj_share / obj_area          attention mass on manipulated-object tokens vs their area share (1 = blind)
  contact_vs_arm                lift of the arm tokens that touch the object over the rest of the arm (1 = no preference)
  inter_vs_rest                 lift of interaction-zone tokens over all other tokens
  top5_precision_inter/obj/body what the model's own top-5% tokens are (fraction that are interaction / object / body)
  top5_mass                     mass concentrated in the top-5% tokens (uniform = 0.05)
  sink_corner_share             mass on the 4 corner tokens of each frame (area 4/80 = 0.05)
  event_frame_shift             (event windows) mass on the latent frame containing the first key event minus mass
                                on the other frame, compared with quiet windows -> temporal focus
  corr_dc / corr_mag            token-level correlation with GT coupling transition / motion magnitude

Output: <model>/analysis/focus.json + printed table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_openloop import window_gt  # noqa: E402

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
GROUPS = {"early": range(0, 10), "mid": range(10, 20), "late": range(20, 30), "all": range(0, 30)}


def lift(att, w):
    a = att / max(att.sum(), 1e-12)
    share = w.sum() / w.size
    return float((a * w).sum() / share) if share > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alpha")
    ap.add_argument("--root", default=str(OUT / "p31"))
    a = ap.parse_args()
    d = Path(a.root) / a.model
    acc = defaultdict(list)
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
        seg = json.loads(str(z["segments"]))
        r = seg["ranges"]
        fv, vh, vw = seg["video_grid"]
        try:
            gt = window_gt(meta["key"], meta["start"])
        except Exception as exc:  # noqa: BLE001
            print("gt failed", meta["key"], exc)
            continue
        A_all = z["attn_step_layer"]                     # [S, L, K]
        L = A_all.shape[1]
        groups = {"early": range(0, L // 3), "mid": range(L // 3, 2 * L // 3), "late": range(2 * L // 3, L), "all": range(0, L)}
        for gname, layers in groups.items():
            layers = [l for l in layers if l < L]
            if not layers:
                continue
            A = A_all[:, layers].sum(axis=(0, 1))
            vf = A[r["video_future"][0]:r["video_future"][1]].reshape(fv - 1, vh, vw)[:, :8]    # head view [2, 8, 10]
            att = vf / max(vf.sum(), 1e-12)
            body = gt["video"]["body"] > 0.2
            obj = (gt["video"]["obj"] > 0.2) & ~body
            manip = (gt["video"]["moving"] > 0.05) & ~body
            inter = gt["video"]["inter"] > 0
            contact_arm = body & inter
            rest_arm = body & ~inter
            k = f"{gname}"
            # object blindness
            if manip.sum():
                acc[f"{k}/manip_share"].append(float(att[manip].sum()))
                acc[f"{k}/manip_area"].append(float(manip.mean()))
                acc[f"{k}/manip_lift"].append(lift(vf, manip.astype(np.float32)))
            if obj.sum():
                acc[f"{k}/obj_lift"].append(lift(vf, obj.astype(np.float32)))
            # contact part of the arm vs the rest of the arm
            if contact_arm.sum() and rest_arm.sum():
                acc[f"{k}/contact_vs_arm"].append(float((att[contact_arm].mean()) / max(att[rest_arm].mean(), 1e-12)))
            if inter.sum() and (~inter).sum():
                acc[f"{k}/inter_vs_rest"].append(float(att[inter].mean() / max(att[~inter].mean(), 1e-12)))
            # what the model's own top-5% tokens are
            flat = att.reshape(-1)
            top = np.argsort(-flat)[: max(1, int(round(0.05 * flat.size)))]
            acc[f"{k}/top5_mass"].append(float(flat[top].sum()))
            for nm, m in (("inter", inter), ("obj", obj | manip), ("body", body)):
                acc[f"{k}/top5_frac_{nm}"].append(float(m.reshape(-1)[top].mean()))
                acc[f"{k}/area_{nm}"].append(float(m.mean()))
            # corner sink
            corners = np.zeros_like(att, dtype=bool)
            corners[:, [0, -1], :][:, :, [0, -1]] = True
            corners[:, 0, 0] = corners[:, 0, -1] = corners[:, -1, 0] = corners[:, -1, -1] = True
            acc[f"{k}/corner_share"].append(float(att[corners].sum()))
            acc[f"{k}/corner_area"].append(float(corners.mean()))
            # temporal: event frame vs other frame
            fm = att.sum(axis=(1, 2))                      # [2]
            events = [e["frame"] for e in meta["events"] if e["type"] in ("contact_start", "motion_onset", "liftoff", "settle")]
            if meta["kind"] == "event" and events:
                ev = min(events)
                ef = 0 if ev <= meta["start"] + 16 else 1
                acc[f"{k}/event_frame_shift"].append(float(fm[ef] - fm[1 - ef]))
                acc[f"{k}/frame2_mass_event_in_f{ef + 1}"].append(float(fm[1]))
            else:
                acc[f"{k}/quiet_frame1_minus_frame2"].append(float(fm[0] - fm[1]))
                acc[f"{k}/frame2_mass_quiet"].append(float(fm[1]))
            for nm in ("dc", "mag"):
                x, y = gt[nm].reshape(-1), att.reshape(-1)
                if x.std() > 0 and y.std() > 0:
                    acc[f"{k}/corr_{nm}"].append(float(np.corrcoef(x, y)[0, 1]))
            ent = -(flat[flat > 0] * np.log(flat[flat > 0])).sum() / np.log(flat.size)
            acc[f"{k}/entropy"].append(float(ent))
    summary = {k: dict(n=len(v), mean=float(np.nanmean(v)), median=float(np.nanmedian(v))) for k, v in acc.items()}
    out_dir = d / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "focus.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "focus_per_window.npz", **{k.replace("/", "__"): np.asarray(v, dtype=np.float32) for k, v in acc.items()})
    print(f"[{a.model}] focus deficiency metrics (head view, future frames)")
    keys = ["entropy", "manip_share", "manip_area", "manip_lift", "obj_lift", "contact_vs_arm", "inter_vs_rest", "top5_mass",
            "top5_frac_inter", "area_inter", "top5_frac_obj", "area_obj", "top5_frac_body", "area_body", "corner_share", "corner_area",
            "event_frame_shift", "quiet_frame1_minus_frame2", "corr_dc", "corr_mag"]
    print(f"{'metric':28s}" + "".join(f"{g:>12s}" for g in GROUPS))
    for kk in keys:
        vals = [summary.get(f"{g}/{kk}", {}).get("mean", np.nan) for g in GROUPS]
        print(f"{kk:28s}" + "".join(f"{v:12.3f}" for v in vals))


if __name__ == "__main__":
    main()
