"""CPU-only snapshot of append-only pi05 metrics; requires matplotlib, not the model runtime."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    if args.window < 2:
        parser.error("window must be at least 2")
    summary_path = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".png" or args.output.exists() or summary_path.exists():
        parser.error("choose a new .png output; snapshots are not overwritten")

    # A writer may be appending the final line; consume complete records only.
    lines = args.metrics.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]
    if len(rows) < args.window:
        parser.error("not enough completed steps for the requested smoothing window")
    invocation_path = args.metrics.parent / "invocation.json"
    total_steps = 1000
    if invocation_path.exists():
        total_steps = json.loads(invocation_path.read_text(encoding="utf-8"))["requested_steps"]
    if type(total_steps) is not int or total_steps < rows[-1]["step"]:
        parser.error("invocation requested_steps must be an integer covering the completed steps")
    steps = np.array([row["step"] for row in rows])
    losses = np.array([row["loss"] for row in rows], dtype=float)
    if not np.isfinite(losses).all() or not np.all(np.diff(steps) == 1):
        parser.error("loss must be finite and steps contiguous; inspect the source log")
    smooth = np.convolve(losses, np.ones(args.window) / args.window, mode="valid")
    smooth_steps = steps[args.window - 1 :]
    first_mean, last_mean = float(losses[: args.window].mean()), float(losses[-args.window :].mean())
    blocks = [
        {
            "from_step": int(steps[index]),
            "to_step": int(steps[min(index + args.window, len(rows)) - 1]),
            "mean_loss": float(losses[index : index + args.window].mean()),
        }
        for index in range(0, len(rows), args.window)
    ]
    summary = {
        "scope": "training_flow_matching_loss_including_native_padding",
        "metrics": str(args.metrics.resolve()),
        "step": int(steps[-1]),
        "total_steps": total_steps,
        "snapshot_time_utc": rows[-1]["time_utc"],
        "window": args.window,
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "relative_loss_drop_pct": 100 * (1 - last_mean / first_mean) if first_mean > 0 else None,
        "nonoverlapping_windows": blocks,
    }
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "savefig.facecolor": "#f7f9fc"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#f7f9fc")
    for ax in axes:
        ax.set_facecolor("white")
        ax.plot(steps, losses, color="#90acd0", alpha=0.6, lw=0.9, label="逐步 loss")
        ax.plot(smooth_steps, smooth, color="#006c70", lw=2.5, label=f"{args.window} 步滑动平均")
        ax.grid(axis="y", color="#e2e7ee", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#c0cad6")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Flow-matching loss")
    axes[0].set_xlim(max(0, steps[0] - 1), steps[-1])
    axes[0].set_ylim(bottom=0)
    axes[0].axvspan(0, 100, color="#dfe9f7", alpha=0.3)
    axes[0].set_title("全程：初期快速下降", loc="left", pad=12)
    axes[0].legend(frameon=False, loc="upper right")
    tail_start = max(int(steps[0]), int(steps[-1]) - 249)
    tail = steps >= tail_start
    axes[1].set_xlim(tail_start, steps[-1])
    axes[1].set_ylim(0, float(losses[tail].max()) * 1.15)
    axes[1].set_title("最近 250 步：放大后期波动", loc="left", pad=12)
    axes[1].scatter([steps[-1]], [smooth[-1]], color="#006c70", s=32, zorder=5)
    axes[1].annotate(
        f"当前均线 {smooth[-1]:.4f}",
        (steps[-1], smooth[-1]),
        xytext=(-12, 22),
        textcoords="offset points",
        ha="right",
        color="#006c70",
        arrowprops={"arrowstyle": "-", "color": "#006c70"},
    )
    stamp = datetime.fromisoformat(rows[-1]["time_utc"]).astimezone().strftime("%m-%d %H:%M:%S %Z")
    fig.suptitle(f"π0.5 关节 SFT · Loss 实时快照 · 第 {steps[-1]} / {total_steps} 步", x=0.07, ha="left", fontsize=17)
    fig.text(
        0.07,
        0.865,
        f"前 {args.window} 步均值 {first_mean:.4f}  →  最近 {args.window} 步 {last_mean:.4f}"
        f"    |    5 条示教 · batch 8    |    {stamp}",
        color="#4c5b70",
    )
    fig.text(
        0.07,
        0.035,
        "左右图纵轴尺度不同；原生训练 loss 含末尾补齐及兼容维度。曲线下降不等于真机任务完成。",
        fontsize=10,
        color="#526175",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.17, wspace=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plot": str(args.output.resolve()), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
