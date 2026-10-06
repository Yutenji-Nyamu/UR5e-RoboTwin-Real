#!/usr/bin/env python3
"""Paper figures / tables for the baseline probing study (reads the analysis outputs, no model inference).

Outputs to ``<probes>/figures/``:
  fig_focus_space.{pdf,png}     spatial focus deficiency: lift per region + composition of the top-5% attended tokens
  fig_focus_time.{pdf,png}      temporal focus deficiency: event-frame attention shift (event vs quiet windows)
  fig_retention.{pdf,png}       P2.2: action deviation vs retained future tokens, structured vs random selection
  fig_world_swap.{pdf,png}      P3.1: action deviation under future-world interventions, per model
  fig_representation.{pdf,png}  P1.1 linear probes, P1.2 region PSNR vs copy-present, P1.3 VAE oracle
  tab_*.csv / tab_*.tex         the numbers behind the figures
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes")
FIG = OUT / "figures"
MODELS = [("alpha", "OpenWAM-Alpha"), ("xwam", "X-WAM"), ("flowwam", "FlowWAM"), ("effwam", "Efficient-WAM")]
UNIT = {"alpha": "cm", "xwam": "cm", "flowwam": "deg", "effwam": "deg", "fastwam": "deg"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 130})


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def available(kind: str):
    out = []
    for key, name in MODELS:
        if kind == "focus" and (OUT / "p31" / key / "analysis" / "focus.json").exists():
            out.append((key, name))
        if kind == "p31" and (OUT / "p31" / key / "analysis" / "conditions.json").exists():
            out.append((key, name))
        if kind == "p22" and (OUT / "p22" / key / "analysis" / "retention.csv").exists():
            out.append((key, name))
    return out


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(FIG / f"{name}.png")


def write_table(name, header, rows):
    with open(FIG / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open(FIG / f"{name}.tex", "w") as fh:
        fh.write("\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n")
        fh.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(str(x) for x in r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------------------- spatial focus


def fig_focus_space():
    models = available("focus")
    if not models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    regions = [("manip_lift", "manipulated\nobject"), ("obj_lift", "all\nobjects"), ("inter_vs_rest", "interaction\nzone vs rest"),
               ("contact_vs_arm", "contact part\nof arm vs\nrest of arm")]
    x = np.arange(len(regions))
    wd = 0.8 / len(models)
    rows = []
    for i, (key, name) in enumerate(models):
        f = load_json(OUT / "p31" / key / "analysis" / "focus.json")
        vals = [f.get(f"all/{r}", {}).get("mean", np.nan) for r, _ in regions]
        axes[0].bar(x + (i - (len(models) - 1) / 2) * wd, vals, wd, label=name)
        rows.append([name] + [f"{v:.2f}" for v in vals] + [f"{f['all/entropy']['mean']:.3f}", f"{f['all/corr_dc']['mean']:.2f}"])
    axes[0].axhline(1, color="k", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n for _, n in regions])
    axes[0].set_ylabel("attention mass / area share  (1 = uniform)")
    axes[0].set_title("(a) Where the action's future-video attention goes")
    axes[0].legend(frameon=False)
    # composition of the top-5% attended tokens vs area composition
    cats = [("obj", "objects"), ("inter", "interaction zone"), ("body", "arm (rest)")]
    labels, comp, area = [], [], []
    for key, name in models:
        f = load_json(OUT / "p31" / key / "analysis" / "focus.json")
        obj = f["all/top5_frac_obj"]["mean"]
        inter = f["all/top5_frac_inter"]["mean"]
        body = max(f["all/top5_frac_body"]["mean"] - inter, 0)
        comp.append([obj, inter, body, max(1 - obj - inter - body, 0)])
        a_obj, a_inter, a_body = f["all/area_obj"]["mean"], f["all/area_inter"]["mean"], max(f["all/area_body"]["mean"] - f["all/area_inter"]["mean"], 0)
        area.append([a_obj, a_inter, a_body, max(1 - a_obj - a_inter - a_body, 0)])
        labels.append(name)
    comp, area = np.array(comp), np.array(area)
    y = np.arange(len(labels))
    colors = ["tab:green", "tab:cyan", "tab:red", "lightgray"]
    names = ["objects", "interaction zone", "arm (rest)", "background"]
    for arr, off, alpha in ((area, -0.2, 0.45), (comp, 0.2, 1.0)):
        left = np.zeros(len(labels))
        for j in range(4):
            axes[1].barh(y + off, arr[:, j], 0.38, left=left, color=colors[j], alpha=alpha, label=names[j] if alpha == 1.0 else None)
            left += arr[:, j]
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("fraction  (upper bar: model's top-5% tokens; lower, faded: area)")
    axes[1].set_title("(b) What the top-5% attended tokens are")
    axes[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7)
    axes[1].set_xlim(0, 1)
    savefig(fig, "fig_focus_space")
    write_table("tab_focus_space", ["model", "manip. object lift", "objects lift", "interaction vs rest", "contact vs arm", "norm. entropy", "corr |dc|"], rows)


def fig_focus_time():
    models = available("focus")
    if not models:
        return
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.0))
    data, pos, labels, colors = [], [], [], []
    rows = []
    for i, (key, name) in enumerate(models):
        z = np.load(OUT / "p31" / key / "analysis" / "focus_per_window.npz")
        groups = [("all__frame2_mass_quiet", "no event", "lightgray"), ("all__frame2_mass_event_in_f1", "event in frame 1", "tab:blue"),
                  ("all__frame2_mass_event_in_f2", "event in frame 2", "tab:orange")]
        for j, (k, lab, col) in enumerate(groups):
            v = z[k] if k in z.files else np.zeros(0)
            data.append(v)
            pos.append(i * 4 + j)
            colors.append(col)
        labels.append(name)
        rows.append([name] + [f"{np.median(z[k]):.3f} (n={len(z[k])})" if k in z.files and len(z[k]) else "-" for k, _, _ in groups])
    bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
    ax.set_xticks([i * 4 + 1 for i in range(len(models))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of future-video attention on frame 2")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in groups], frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Temporal focus: attention on the later frame does not depend on\nwhether the interaction event falls in frame 1 or frame 2")
    savefig(fig, "fig_focus_time")
    write_table("tab_focus_time", ["model", "frame-2 share: no event", "event in frame 1", "event in frame 2"], rows)


# ----------------------------------------------------------------------------- retention


def fig_retention():
    models = available("p22")
    if not models:
        return
    rules = [("random", "random", "k"), ("inter", "interaction zone (GT)", "tab:cyan"), ("dc", "coupling transition |Δc| (GT)", "tab:purple"),
             ("mag", "motion magnitude (GT)", "tab:orange"), ("obj", "object tokens (GT)", "tab:green")]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.2), squeeze=False)
    rows = []
    for ax, (key, name) in zip(axes[0], models):
        tab = {r["condition"]: r for r in csv.DictReader(open(OUT / "p22" / key / "analysis" / "retention.csv"))}
        for rule, lab, col in rules:
            xs, ys = [], []
            for fr in (20, 5, 1):
                c = f"{rule}_{fr}"
                if c in tab:
                    xs.append(fr)
                    ys.append(float(tab[c]["dev_pos_cm_mean"]))
            ax.plot(xs, ys, marker="o", color=col, label=lab)
            rows.append([name, lab] + [f"{v:.2f}" for v in ys])
        for ref, ls, lab in (("head_only", "--", "all head tokens (100%)"), ("drop_world", ":", "no future tokens (0%)")):
            if ref in tab:
                ax.axhline(float(tab[ref]["dev_pos_cm_mean"]), ls=ls, color="gray", label=lab)
        ax.set_xscale("log")
        ax.set_xticks([20, 5, 1])
        ax.set_xticklabels(["20%", "5%", "1%"])
        ax.invert_xaxis()
        ax.set_xlabel("future tokens kept for the action (head view)")
        ax.set_ylabel(f"action deviation vs full ({UNIT[key]})")
        ax.set_title(f"{name}  (n={tab['full']['n']})")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("P2.2  Keeping the physically relevant tokens is no better than keeping random ones", y=1.02)
    savefig(fig, "fig_retention")
    write_table("tab_retention", ["model", "selection rule", "20%", "5%", "1%"], rows)


# ----------------------------------------------------------------------------- world replacement


def fig_world_swap():
    models = available("p31")
    if not models:
        return
    conds = [("swap_static_video", "static future"), ("swap_shift_video", "time-shifted"), ("pool_video", "pooled"), ("drop_video", "dropped"),
             ("swap_same_task_video", "other episode\n(same task)"), ("swap_other_task_video", "other task")]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.2), squeeze=False)
    rows = []
    for ax, (key, name) in zip(axes[0], models):
        cj = load_json(OUT / "p31" / key / "analysis" / "conditions.json")
        M = (OUT / "p31" / key / "analysis" / "primary_metric.txt").read_text().strip() if (OUT / "p31" / key / "analysis" / "primary_metric.txt").exists() else "pos_cm"
        means = [cj[c][f"vs_full_{M}_all"]["mean"] for c, _ in conds]
        meds = [cj[c][f"vs_full_{M}_all"]["median"] for c, _ in conds]
        gt = cj["full"][f"vs_gt_{M}_all"]["mean"]
        x = np.arange(len(conds))
        ax.bar(x, means, color="tab:red", alpha=0.75, label="mean")
        ax.plot(x, meds, "k_", ms=14, mew=2, label="median")
        ax.axhline(gt, color="tab:blue", ls="--", lw=1, label=f"full model vs GT ({gt:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in conds], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(f"action deviation vs full ({UNIT[key]})")
        ax.set_title(f"{name}  (n={cj['full'][f'vs_full_{M}_all']['n']})")
        ax.legend(frameon=False, fontsize=7)
        rows.append([name, UNIT[key], f"{gt:.2f}"] + [f"{m:.2f} / {d:.2f}" for m, d in zip(means, meds)])
    fig.suptitle("P3.1  Replacing what the action reads from the predicted future (open loop)", y=1.02)
    savefig(fig, "fig_world_swap")
    write_table("tab_world_swap", ["model", "unit", "full vs GT"] + [l.replace("\n", " ") for _, l in conds], rows)


# ----------------------------------------------------------------------------- representation


def fig_representation():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    # (a) linear probes
    lp = load_json(OUT / "p22" / "alpha" / "analysis" / "linear_probe.json")
    if lp:
        targets = [("body", "arm present"), ("obj", "object present"), ("inter", "interaction zone"), ("moving", "object moving"), ("dc", "coupling |Δc|"), ("mag", "motion magnitude")]
        layers = sorted(lp, key=lambda k: int(k.split("_L")[-1]))
        x = np.arange(len(targets))
        wd = 0.8 / len(layers)
        for i, layer in enumerate(layers):
            vals = [lp[layer].get(t, {}).get("r2", np.nan) for t, _ in targets]
            axes[0].bar(x + (i - (len(layers) - 1) / 2) * wd, vals, wd, label=f"OpenWAM-Alpha video tokens, {layer.split('_')[-1]}")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([l for _, l in targets], rotation=25, ha="right", fontsize=7)
        axes[0].set_ylabel("linear-probe R² (grouped 5-fold)")
        axes[0].set_title("(a) What the future-video tokens encode")
        axes[0].legend(frameon=False, fontsize=7)
        axes[0].set_ylim(0, 1)
    # (b) region PSNR vs copy-present
    p12 = load_json(OUT / "p31" / "alpha" / "analysis" / "p12_regions.json")
    if p12:
        regs = [("body", "arm"), ("object", "objects"), ("moving_object", "moving objects"), ("background", "background")]
        x = np.arange(len(regs))
        pred = [p12[f"psnr_{r}"]["median"] for r, _ in regs]
        copy = [p12.get(f"psnr_copy_{r}", {}).get("median", np.nan) for r, _ in regs]
        axes[1].bar(x - 0.2, pred, 0.4, label="predicted future (Alpha)", color="tab:red")
        axes[1].bar(x + 0.2, copy, 0.4, label="copy current frame", color="gray")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([l for _, l in regs])
        axes[1].set_ylabel("PSNR vs GT future (dB, median)")
        axes[1].set_title("(b) Region-wise quality of the predicted future")
        axes[1].legend(frameon=False, fontsize=7)
    # (c) VAE oracle
    vo = load_json(OUT / "vae_oracle" / "summary.json")
    if vo:
        t = vo["table"]
        bins = t["vae/object"]["bins_m"]
        xs = [f"{int(bins[i]*1000)}–{int(bins[i+1]*1000)}" for i in range(len(bins) - 2)]
        for stage, ls in (("codec", ":"), ("video", "--"), ("vae", "-")):
            for role, col in (("body", "tab:red"), ("object", "tab:green")):
                axes[2].plot(xs, t[f"{stage}/{role}"]["epe_mm"][:-1], ls=ls, color=col, marker="o", ms=3,
                             label=f"{role}, {'μ-law' if stage=='codec' else '+h264' if stage=='video' else '+Wan VAE'}")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("GT displacement (mm)")
        axes[2].set_ylabel("round-trip EPE (mm)")
        axes[2].set_title("(c) Track4D-as-RGB through the video VAE")
        axes[2].legend(frameon=False, fontsize=6, ncol=2)
        axes[2].tick_params(axis="x", labelsize=7)
    savefig(fig, "fig_representation")


def main() -> None:
    fig_focus_space()
    fig_focus_time()
    fig_retention()
    fig_world_swap()
    fig_representation()


if __name__ == "__main__":
    main()
