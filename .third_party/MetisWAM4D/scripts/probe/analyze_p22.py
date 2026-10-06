#!/usr/bin/env python3
"""Aggregate the P2.2 token-retention pass (and the P1.1 linear probes on the captured features).

P2.2: for every keep-subset condition, action deviation from the un-intervened run (cm / deg / gripper)
and error vs. GT -> ``retention.csv`` + ``retention.png`` (deviation vs. retained fraction, one curve per
selection rule, ``head_only`` and ``drop_world`` as references).

P1.1: ridge regression from the future-world token features (value projections at layers 5/15/25, last
denoising step) to GT token quantities (object motion magnitude, coupling transition |dc|, interaction-zone
fraction, body fraction, object fraction).  Video features (both models) vs. Track features (Janus), grouped
5-fold CV by episode, R^2 (regression) -> ``linear_probe.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
RULES = ("random", "inter", "dc", "mag", "obj")
FRACS = (20, 5, 1)


def summarize(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                p75=float(np.percentile(v, 75)) if len(v) else None)


def retention(files: list[Path], out_dir: Path, model: str) -> dict:
    rows = defaultdict(lambda: defaultdict(list))
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
        errors = json.loads(str(z["errors"]))
        for cond, e in errors.items():
            for m in ("pos_cm", "rot_deg", "grip", "joint_deg"):
                rows[cond][f"vs_full_{m}"].append(e["vs_full"].get(m, float("nan")))
                rows[cond][f"vs_full_{m}_{meta['kind']}"].append(e["vs_full"].get(m, float("nan")))
                rows[cond][f"vs_gt_{m}"].append(e["vs_gt"].get(m, float("nan")))
    # joint-space models (Efficient-WAM / FlowWAM) have no EEF error: report joint degrees under the pos_cm columns
    if rows and not any(np.isfinite(v) for v in rows.get("full", next(iter(rows.values())))["vs_gt_pos_cm"]):
        print(f"[{model}] no EEF metric, primary deviation metric = joint_deg")
        for cond in rows:
            for k in list(rows[cond]):
                if "pos_cm" in k:
                    rows[cond][k] = rows[cond][k.replace("pos_cm", "joint_deg")]
    table = {c: {k: summarize(v) for k, v in m.items()} for c, m in rows.items()}
    with open(out_dir / "retention.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "n", "dev_pos_cm_mean", "dev_pos_cm_median", "dev_rot_deg_mean", "dev_grip_mean", "vs_gt_pos_cm_mean", "event_dev_pos", "quiet_dev_pos"])
        for c, m in sorted(table.items(), key=lambda kv: kv[1]["vs_full_pos_cm"]["mean"] or 0):
            w.writerow([c, m["vs_full_pos_cm"]["n"], m["vs_full_pos_cm"]["mean"], m["vs_full_pos_cm"]["median"], m["vs_full_rot_deg"]["mean"],
                        m["vs_full_grip"]["mean"], m["vs_gt_pos_cm"]["mean"], m.get("vs_full_pos_cm_event", {}).get("mean"), m.get("vs_full_pos_cm_quiet", {}).get("mean")])
    print(f"\n[{model}] P2.2 retention ({len(files)} windows): condition  dev_pos_cm mean / median   rot  grip | vsGT")
    for c, m in sorted(table.items(), key=lambda kv: kv[1]["vs_full_pos_cm"]["mean"] or 0):
        x = m["vs_full_pos_cm"]
        fmt = lambda v: f"{v:6.2f}" if v is not None else "   nan"  # noqa: E731
        print(f"  {c:14s} {fmt(x['mean'])} / {fmt(x['median'])}  {fmt(m['vs_full_rot_deg']['mean'])} {fmt(m['vs_full_grip']['mean'])} | {fmt(m['vs_gt_pos_cm']['mean'])}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, stat in zip(axes, ("mean", "median")):
            for rule in RULES:
                xs, ys = [], []
                for fr in FRACS:
                    c = f"{rule}_{fr}"
                    if c in table:
                        xs.append(fr)
                        ys.append(table[c]["vs_full_pos_cm"][stat])
                ax.plot(xs, ys, marker="o", label=rule)
            for ref, ls in (("head_only", "--"), ("drop_world", ":"), ("frame1_only", "-."), ("frame2_only", "-.")):
                if ref in table:
                    ax.axhline(table[ref]["vs_full_pos_cm"][stat], ls=ls, color="gray", label=ref)
            ax.set_xscale("log")
            ax.set_xlabel("retained future world tokens (%)")
            ax.set_ylabel(f"action deviation vs full ({stat}, cm)")
            ax.set_title(f"{model}: {stat}")
            ax.invert_xaxis()
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "retention.png", dpi=130)
    except Exception as exc:  # noqa: BLE001
        print("plot skipped:", exc)
    return table


def linear_probe(files: list[Path], out_dir: Path, model: str) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    targets = {
        "video": {"mag": "gt_mag", "dc": "gt_dc", "inter": "gt_video_inter", "body": "gt_video_body", "obj": "gt_video_obj", "moving": "gt_video_moving"},
        "track": {"mag": "gt_mag", "dc": "gt_dc", "inter": "gt_track_inter", "body": "gt_track_body", "obj": "gt_track_obj", "moving": "gt_track_moving"},
    }
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    groups: dict[str, list] = defaultdict(list)
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = z["meta"].item() if z["meta"].dtype == object else json.loads(str(z["meta"]))
        seg = json.loads(str(z["segments"]))
        for key in z.files:
            if not key.startswith("feat_"):
                continue
            _, segname, rest = key.split("_", 2)          # feat_video_future_L5 -> segname 'video', rest 'future_L5'
            layer = rest.split("_L")[-1]
            X = z[key].astype(np.float32)                   # [N_tokens, 3072]
            if segname == "video":
                f_, h, w = seg["video_grid"]
                X = X.reshape(f_ - 1, h, w, -1)[:, :8].reshape(-1, X.shape[-1])     # head view only
            gt_ok = True
            y = {}
            for tname, gkey in targets[segname].items():
                if gkey not in z.files:
                    gt_ok = False
                    break
                y[tname] = z[gkey].astype(np.float32).reshape(-1)
            if not gt_ok:
                continue
            name = f"{segname}_L{layer}"
            data[name]["X"].append(X)
            for tname, arr in y.items():
                data[name][tname].append(arr)
            groups[name].extend([meta["key"].rsplit("/", 1)[0] + "/" + meta["key"].rsplit("/", 1)[1]] * len(X))
    results = {}
    for name, d in data.items():
        X = np.concatenate(d["X"])
        g = np.asarray(groups[name])
        X = PCA(n_components=min(256, X.shape[1], X.shape[0] - 1), random_state=0).fit_transform(X)
        res = {}
        for tname in targets["video"]:
            if tname not in d:
                continue
            y = np.concatenate(d[tname])
            if y.std() < 1e-8:
                continue
            r2s, r2_shuf = [], []
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                m = Ridge(alpha=10.0).fit(X[tr], y[tr])
                pred = m.predict(X[te])
                r2s.append(1 - ((pred - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
                ys = np.random.default_rng(0).permutation(y[tr])
                m2 = Ridge(alpha=10.0).fit(X[tr], ys)
                p2 = m2.predict(X[te])
                r2_shuf.append(1 - ((p2 - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
            res[tname] = dict(r2=float(np.mean(r2s)), r2_shuffled=float(np.mean(r2_shuf)), n=int(len(y)))
        results[name] = res
    (out_dir / "linear_probe.json").write_text(json.dumps(results, indent=1))
    print(f"\n[{model}] P1.1 linear probes (R^2, grouped 5-fold, PCA-256 features):")
    for name, res in sorted(results.items()):
        print("  " + name.ljust(16) + " ".join(f"{t}={v['r2']:.3f}" for t, v in res.items()))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="janus")
    ap.add_argument("--root", default=str(OUT / "p22"))
    a = ap.parse_args()
    d = Path(a.root) / a.model
    files = sorted(d.glob("*.npz"))
    out_dir = d / "analysis"
    out_dir.mkdir(exist_ok=True)
    retention(files, out_dir, a.model)
    linear_probe(files, out_dir, a.model)


if __name__ == "__main__":
    main()
