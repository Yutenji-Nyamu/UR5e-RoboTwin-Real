"""Window geometry shared by the human datasets (OpenWAM-Alpha contract, 30 fps sources).

    video   frames s + 4k, k = 0..8                 9 frames
    track   transitions s + 4k -> s + 4k + 4, k = 0..7, plus the zero-displacement anchor at s
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from PIL import Image

SOURCE_FRAMES = 33
STRIDE = 4
VIDEO_FRAMES = 9
TRANSITIONS = 8
FPS = 30
WIDTH, HEIGHT = 512, 288


def video_frame_indices(start: int) -> list[int]:
    return [start + STRIDE * k for k in range(VIDEO_FRAMES)]


def source_frame_map(n_out: int, src_fps: str | float, src_frames: int, out_fps: int = FPS) -> np.ndarray:
    """30 fps frame ``k`` <- source frame ``round(k * src_fps / 30)`` (the transcoder's ``fps=`` filter)."""
    f = float(Fraction(str(src_fps)))
    return np.minimum(np.rint(np.arange(n_out) * f / out_fps).astype(np.int64), max(src_frames - 1, 0))


def fit_frame(rgb: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize (no-op for native frames)."""
    h, w = rgb.shape[:2]
    if (w, h) == (width, height):
        return rgb
    target = width / height
    if w / h > target:
        new_w = int(round(h * target))
        x0 = (w - new_w) // 2
        rgb = rgb[:, x0:x0 + new_w]
    elif w / h < target:
        new_h = int(round(w / target))
        y0 = (h - new_h) // 2
        rgb = rgb[y0:y0 + new_h]
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8)


__all__ = [
    "FPS", "HEIGHT", "SOURCE_FRAMES", "STRIDE", "TRANSITIONS", "VIDEO_FRAMES", "WIDTH", "fit_frame",
    "source_frame_map", "video_frame_indices",
]
