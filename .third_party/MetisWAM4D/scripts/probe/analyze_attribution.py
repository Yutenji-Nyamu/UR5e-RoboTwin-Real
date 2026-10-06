#!/usr/bin/env python3
"""Aggregate the attribution pass: attention vs attention x gradient vs latent saliency, same lifts / composition.

Also makes ``fig_three_lenses``: per lens the lift on interaction zone / manipulated object / arm-contact / arm-rest,
and a per-window scatter (object attention share vs object area share) for all lenses.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_openloop import window_gt  # noqa: E402

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
LENSES = [("attention", "attention"), ("attn_grad", "attention × gradient"), ("latent_saliency", "input saliency (∂action/∂latent)")]


def lift(m, w):
    m = m / max(m.sum(), 1e-12)
    return float((m * w).sum() / w.mean()) if w.mean() > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alpha")
    a = ap.parse_args()
    d = OUT / "attribution" / a.model
    acc = defaultdict(list)
    scat = defaultdict(list)
    n = 0
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        seg = json.loads(str(z["segments"]))
        r = seg["ranges"]
        fv, vh, vw = seg["video_grid"]
        try:
            gt = window_gt(meta["key"], meta["start"])
        except Exception:  # noqa: BLE001
            continue
        inter = gt["video"]["inter"] > 0
        manip = gt["video"]["moving"] > 0.05
        body = gt["video"]["body"] > 0.2
        obj = gt["video"]["obj"] > 0.2
        contact, rest = body & inter, body & ~inter
        for s in z["steps"]:
            maps = {
                "attention": z[f"attention_s{s}"][r["video_future"][0]:r["video_future"][1]].reshape(fv - 1, vh, vw)[:, :8],
                "attn_grad": np.abs(z[f"attn_grad_s{s}"][r["video_future"][0]:r["video_future"][1]]).reshape(fv - 1, vh, vw)[:, :8],
                "latent_saliency": z[f"latent_saliency_s{s}"][:, :8],
            }
            for lens, m in maps.items():
                m = np.maximum(m, 0)
                for nm, w in (("inter", inter), ("manip", manip), ("obj", obj), ("body", body)):
                    if w.any():
                        acc[f"{lens}/s{s}/{nm}_lift"].append(lift(m, w.astype(np.float32)))
                if contact.any() and rest.any():
                    p = m / max(m.sum(), 1e-12)
                    acc[f"{lens}/s{s}/contact_vs_arm"].append(float(p[contact].mean() / max(p[rest].mean(), 1e-12)))
                flat = (m / max(m.sum(), 1e-12)).reshape(-1)
                top = np.argsort(-flat)[: max(1, int(round(0.05 * flat.size)))]
                acc[f"{lens}/s{s}/top5_frac_obj"].append(float((obj | manip).reshape(-1)[top].mean()))
                acc[f"{lens}/s{s}/top5_frac_inter"].append(float(inter.reshape(-1)[top].mean()))
                acc[f"{lens}/s{s}/top5_frac_body"].append(float(body.reshape(-1)[top].mean()))
                if manip.any() and s == z["steps"][len(z["steps"]) // 2]:
                    scat[lens].append((float(manip.mean()), float((m / max(m.sum(), 1e-12))[manip].sum())))
        n += 1
    summary = {k: dict(n=len(v), mean=float(np.nanmean(v)), median=float(np.nanmedian(v))) for k, v in acc.items()}
    out_dir = d / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "attribution.json").write_text(json.dumps(summary, indent=1))
    steps = sorted({int(k.split("/")[1][1:]) for k in acc})
    print(f"[{a.model}] attribution over {n} windows (mean lift; 1 = uniform)")
    print(f"{'lens/step':36s} {'inter':>7s} {'manip':>7s} {'objects':>8s} {'arm':>6s} {'contact/arm':>12s} | top5%: obj  inter  arm")
    for lens, lab in LENSES:
        for s in steps:
            g = lambda k: summary.get(f"{lens}/s{s}/{k}", {}).get("mean", np.nan)  # noqa: E731
            print(f"{lab + f' @step {s}':36s} {g('inter_lift'):7.2f} {g('manip_lift'):7.2f} {g('obj_lift'):8.2f} {g('body_lift'):6.2f} {g('contact_vs_arm'):12.2f} | "
                  f"{g('top5_frac_obj'):5.2f} {g('top5_frac_inter'):6.2f} {g('top5_frac_body'):5.2f}")
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    regions = [("inter_lift", "interaction\nzone"), ("manip_lift", "manipulated\nobject"), ("obj_lift", "all\nobjects"), ("contact_vs_arm", "contact part\nvs rest of arm"), ("body_lift", "arm")]
    x = np.arange(len(regions))
    wd = 0.8 / len(LENSES)
    mid = steps[len(steps) // 2]
    for i, (lens, lab) in enumerate(LENSES):
        vals = [summary.get(f"{lens}/s{mid}/{k}", {}).get("mean", np.nan) for k, _ in regions]
        axes[0].bar(x + (i - 1) * wd, vals, wd, label=lab)
    axes[0].axhline(1, color="k", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([l for _, l in regions], fontsize=8)
    axes[0].set_ylabel("mass / area share  (1 = uniform)")
    axes[0].set_title(f"(a) Three lenses agree (OpenWAM-Alpha, denoising step {mid}, n={n})")
    axes[0].legend(frameon=False, fontsize=7)
    for lens, lab in LENSES:
        pts = np.array(scat[lens])
        if len(pts):
            axes[1].scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.5, label=lab)
    lim = max(0.05, max((np.array(v)[:, 0].max() for v in scat.values() if len(v)), default=0.2) * 1.05)
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.8, label="= area share (blind)")
    axes[1].set_xlabel("area share of the manipulated object (head view)")
    axes[1].set_ylabel("share of mass on the manipulated object")
    axes[1].set_title("(b) Per window: mass on the object vs its area")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim * 1.6)
    fig.tight_layout()
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "fig_three_lenses.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_three_lenses.png", bbox_inches="tight", dpi=170)
    print(fig_dir / "fig_three_lenses.png")


if __name__ == "__main__":
    main()
