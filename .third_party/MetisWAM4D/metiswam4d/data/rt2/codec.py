"""Pixel-frame Track4D codec: (du/W, dv/W, dd) -> signed mu-law RGB.

A surface point seen at pixel ``(u, v)`` with camera depth ``d`` in source frame ``t`` is re-observed at
``t + 4``; its position is expressed in the camera-``t`` frame (ego-motion compensated) and projected with the
camera-``t`` intrinsics.  The three channels are

    u = du / W      image displacement, fraction of the frame width
    v = dv / W      image displacement, fraction of the frame width (same denominator: isotropic in
                    pixels, independent of the aspect ratio, so one colour means one motion on 16:9 and 4:3 sources)
    d = dd          camera-depth change in metres

Before encoding, each channel is soft-shrunk by its noise floor (``|x| -> max(|x| - tau, 0)``, sign kept):
0.5 px on u/v (RAFT / rasteriser jitter) and a dataset-dependent ``tau_d`` on d (DA3 frame-to-frame depth
noise 8 mm on IG-10K, MANO jitter 2 mm on Kling, GT 2 mm on RT2).  Then

    z_c = clip(x_c / s_c, -1, 1),  R_c = round(127.5 * (1 + sgn(z_c) log(1 + mu |z_c|) / log(1 + mu)))

with the frozen scales ``s = (1/6 W, 1/6 W, 0.10 m)`` and ``mu = 31``.  128 = zero, invalid pixels are black (0),
the anchor frame is 128 inside the foreground mask and 0 elsewhere.  Statistics behind the constants:
``docs/2026-09-26-Track4D表示-相机系xyz与像素系uvd对比.md``.
"""
from __future__ import annotations

import numpy as np

MU = 31.0
UVD_SCALE = np.asarray((1.0 / 6.0, 1.0 / 6.0, 0.10), dtype=np.float32)   # frame-width fraction (u, v), metres (d)
SHRINK_PX = 0.5                                                           # u / v soft threshold, pixels
SHRINK_D_M = {"ig10k": 0.008, "kling": 0.002, "rt2": 0.002, "robodojo": 0.002}  # d soft threshold, metres
# robodojo: FK-exact displacement; keeps RT2's colour mapping so the RT2-trained Track expert transfers
CODEC_ID = "metiswam4d.uvd_rgb.mu31.s6w_6w_0p10.v2"


def soft_shrink(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def uvd_from_xyz(delta_xyz: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame displacement ``[..., H, W, 3]`` (metres) + source-frame depth ``[..., H, W]`` + intrinsics
    ``K [3, 3]`` -> ``(du px, dv px, dd m)`` ``[..., H, W, 3]``.  Pixels with non-positive depth get zeros."""
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    h, w = depth_m.shape[-2:]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    z0 = np.asarray(depth_m, dtype=np.float32)
    ok = np.isfinite(z0) & (z0 > 0)
    z0 = np.where(ok, z0, 1.0)
    d = np.asarray(delta_xyz, dtype=np.float32)
    x1 = (xx - cx) / fx * z0 + d[..., 0]
    y1 = (yy - cy) / fy * z0 + d[..., 1]
    z1 = np.maximum(z0 + d[..., 2], 1e-3)
    out = np.stack((x1 / z1 * fx + cx - xx, y1 / z1 * fy + cy - yy, d[..., 2]), axis=-1)
    out[~ok] = 0.0
    return out


def normalize_uvd(uvd: np.ndarray, width: int, tau_d: float) -> np.ndarray:
    """``(du px, dv px, dd m)`` -> shrunk, scale-normalised channels in ``[-1, 1]`` (the ``track_delta`` target).
    ``width`` is the frame width in pixels; both image channels are divided by it."""
    u = soft_shrink(uvd[..., 0], SHRINK_PX) / float(width)
    v = soft_shrink(uvd[..., 1], SHRINK_PX) / float(width)
    d = soft_shrink(uvd[..., 2], tau_d)
    return np.clip(np.stack((u, v, d), axis=-1).astype(np.float32) / UVD_SCALE, -1.0, 1.0)


def mu_law(z: np.ndarray, mu: float = MU) -> np.ndarray:
    """Normalised ``[-1, 1]`` -> uint8 (128 = zero)."""
    enc = np.sign(z) * np.log1p(mu * np.abs(z)) / np.log1p(mu)
    return np.round(127.5 * (1.0 + enc)).astype(np.uint8)


def inverse_mu_law(rgb: np.ndarray, mu: float = MU) -> np.ndarray:
    enc = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return np.sign(enc) * (np.expm1(np.abs(enc) * np.log1p(mu)) / mu)


def encode_uvd(uvd: np.ndarray, valid: np.ndarray, width: int, tau_d: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (uint8 RGB ``[..., H, W, 3]`` with invalid pixels black, normalised ``track_delta`` float32)."""
    z = normalize_uvd(uvd, width, tau_d)
    rgb = mu_law(z)
    inval = ~np.asarray(valid, dtype=bool)
    rgb[inval] = 0
    z[inval] = 0.0
    return rgb, z


def anchor_frame(foreground: np.ndarray) -> np.ndarray:
    """Zero-displacement sentinel: 128 on the foreground, 0 elsewhere."""
    out = np.zeros((*foreground.shape, 3), dtype=np.uint8)
    out[np.asarray(foreground, dtype=bool)] = 128
    return out


__all__ = ["CODEC_ID", "MU", "SHRINK_D_M", "SHRINK_PX", "UVD_SCALE", "anchor_frame", "encode_uvd", "inverse_mu_law",
           "mu_law", "normalize_uvd", "soft_shrink", "uvd_from_xyz"]
