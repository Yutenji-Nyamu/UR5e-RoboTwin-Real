"""QC panel for one ``generate_4d.py`` episode: 4 frames x (head RGB + role overlay + (du, dv) arrows | the same
points drawn on frame t+4 at their predicted landing pixels | depth | |du, dv| | dd), plus the EEF10 action curves.

    python scripts/vlabench/qc_panel.py <episode.h5> <out.png>
"""
from __future__ import annotations

import io
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def head(h, t, w, hh):
    return np.asarray(Image.open(io.BytesIO(h["head_rgb"][t].tobytes())).convert("RGB").resize((w, hh), Image.LANCZOS))


def main(path: str, out: str) -> None:
    with h5py.File(path) as h:
        n = int(h.attrs["frames"])
        role, delta, depth = h["role"], h["delta_uvd"], h["head_depth_mm"]
        hh, w = role.shape[1:]
        ts = np.linspace(0, n - 5, 4).astype(int)
        fig, axes = plt.subplots(5, 5, figsize=(25, 20))
        for r, t in enumerate(ts):
            img, img4 = head(h, t, w, hh), head(h, t + 4, w, hh)
            ro, d = role[t], delta[t].astype(np.float32)
            overlay = img.copy()
            overlay[ro == 1] = (0.5 * overlay[ro == 1] + [0, 0, 127]).astype(np.uint8)
            overlay[ro == 2] = (0.5 * overlay[ro == 2] + [127, 0, 0]).astype(np.uint8)
            ys, xs = np.nonzero((ro > 0) & (np.hypot(d[..., 0], d[..., 1]) > 0.5))
            pick = np.random.default_rng(0).choice(len(ys), min(len(ys), 150), replace=False) if len(ys) else []
            ax = axes[r, 0]; ax.imshow(overlay); ax.set_title(f"t={t} role (blue robot, red object) + du,dv")
            for i in pick:
                ax.arrow(xs[i], ys[i], d[ys[i], xs[i], 0], d[ys[i], xs[i], 1], color="yellow", width=0.3, head_width=2)
            ax = axes[r, 1]; ax.imshow(img4); ax.set_title(f"landing on t+4={t + 4}")
            if len(pick):
                ax.scatter(xs[pick] + d[ys[pick], xs[pick], 0], ys[pick] + d[ys[pick], xs[pick], 1], s=4, c="lime")
            axes[r, 2].imshow(depth[t].astype(np.float32), cmap="turbo"); axes[r, 2].set_title("depth mm")
            axes[r, 3].imshow(np.hypot(d[..., 0], d[..., 1]), cmap="magma"); axes[r, 3].set_title("|du,dv| px")
            axes[r, 4].imshow(d[..., 2], cmap="coolwarm", vmin=-0.03, vmax=0.03); axes[r, 4].set_title("dd m")
        for ax in axes[:4].ravel():
            ax.axis("off")
        a = h["actions7"][()]
        s = h["state7"][()]
        for j, name in enumerate(("x", "y", "z")):
            axes[4, 0].plot(a[:, j], label=f"act {name}"); axes[4, 0].plot(s[:, j], "--", label=f"state {name}")
        axes[4, 0].legend(fontsize=7)
        axes[4, 1].plot(a[:, 3:6]); axes[4, 1].set_title("action euler")
        axes[4, 2].plot(a[:, 6], label="action open"); axes[4, 2].plot(s[:, 6], label="state flag"); axes[4, 2].legend()
        for ax in axes[4, 3:]:
            ax.axis("off")
        fig.suptitle(f"{h.attrs['task']}: {h.attrs['instruction']}  frames={n}  "
                     f"landing median {h.attrs['qc_landing_depth_err_median_m'] * 1000:.2f} mm, "
                     f"{h.attrs['qc_landing_within_5mm'] * 100:.1f}% < 5 mm")
    fig.tight_layout()
    fig.savefig(out, dpi=60)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
