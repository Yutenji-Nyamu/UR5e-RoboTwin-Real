"""Hand Track4D for KlingHumanEgo-2.5M-5000H windows, rendered on the fly from the sparse MANO fits.

Per clip the fit archive holds one record per confident WiLoR detection: MANO ``theta`` (48), ``betas``
(10), a weak-perspective camera ``cam = (s, cx, cy)`` (pixels per metre and the wrist pixel), source
``frame_index``, ``track_id``, ``is_right``.  For a window this module

1. runs smplx MANO on the detections at the nine source frames of the window (778 wrist-centred
   vertices per detection; same vertex index = same surface point),
2. places every hand in metric camera space: root depth ``z = f_x / s`` (the metric MANO hand seen at
   ``s`` px/m under the clip's intrinsics) and root ``r = z K^-1 [cx, cy, 1]``.  Root depth is held at the
   source-frame value for the target frame (the root's depth change is not observable from a monocular
   weak-perspective fit; articulation keeps its full XYZ),
3. compensates the head ego-motion with the clip's camera trajectory: the target-frame surface point is
   expressed in the source-frame camera axes, ``X_a(t+4) = R X_b + t``, so the displacement is the hand's
   own motion (Track4D definition; a static hand under a turning head gives zero),
4. projects both endpoints with the source-frame intrinsics and paints every vertex with
   ``(du/W, dv/W, dd)``, mu-law encoded with the shared codec (:mod:`metiswam4d.data.rt2.codec`).

Steps 1-4 run in the DataLoader worker (:func:`hand_window_meshes`, fixed-shape arrays so windows collate).
Rasterisation of the source-frame meshes is :func:`hand_pixels`, batched over the whole batch on the
training GPU (pytorch3d, both hands of an image in one mesh so occlusion is resolved by depth): the
naive CPU rasteriser costs ~2 s per image single-threaded, the GPU ~3 ms.

The camera code of every transition (``T_{t -> t+4}`` in the camera-``t`` frame) is returned alongside
for the Track expert's camera tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from metiswam4d.data.camera import encode_camera_delta
from metiswam4d.data.human.window import STRIDE, TRANSITIONS, VIDEO_FRAMES
from metiswam4d.data.rt2.codec import MU, SHRINK_D_M, encode_uvd

MANO_DIR = Path("/m2v_intern_v3/danglingwei/model_zoos/mano_v1_2/models")


@lru_cache(maxsize=2)
def mano_model(is_right: bool):
    import smplx
    m = smplx.MANO(str(MANO_DIR / ("MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl")), is_rhand=is_right,
                   use_pca=False, flat_hand_mean=True, batch_size=1)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def mano_vertices(is_right: bool, betas: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``betas [N, 10]``, ``theta [N, 48]`` -> wrist-centred vertices ``[N, 778, 3]``."""
    m = mano_model(bool(is_right))
    th = torch.from_numpy(np.ascontiguousarray(theta, dtype=np.float32))
    be = torch.from_numpy(np.ascontiguousarray(betas, dtype=np.float32))
    out = m(betas=be, global_orient=th[:, :3], hand_pose=th[:, 3:], return_verts=True)
    wrist = out.joints[:, :1]
    return (out.vertices - wrist).numpy()


def relative_pose(extrinsics: np.ndarray, a: int, b: int, world_to_cam: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` with ``X_a = R X_b + t``: the camera-``b`` frame expressed in the camera-``a`` frame."""
    Ra, ta = extrinsics[a, :, :3], extrinsics[a, :, 3]
    Rb, tb = extrinsics[b, :, :3], extrinsics[b, :, 3]
    if world_to_cam:          # X_c = R X_w + t
        R = Ra @ Rb.T
        t = ta - R @ tb
    else:                     # X_w = R X_c + t
        R = Ra.T @ Rb
        t = Ra.T @ (tb - ta)
    return R.astype(np.float32), t.astype(np.float32)


MAX_HANDS = 2      # hands per image; WiLoR fits never hold more than two detections per frame
MANO_VERTS = 778


@dataclass
class HandMeshes:
    verts: np.ndarray            # float32 [9, MAX_HANDS, 778, 3]  NDC of the source-frame meshes (image 0 = anchor)
    colors: np.ndarray           # float32 [9, MAX_HANDS, 778, 3]  encoded displacement per vertex (image 0 unused)
    is_right: np.ndarray         # bool [9, MAX_HANDS]
    present: np.ndarray          # bool [9, MAX_HANDS]
    camera_delta: np.ndarray     # float32 [8, 9]
    camera_valid: np.ndarray     # bool [8]
    pairs_per_transition: np.ndarray  # int [8]  tracked hands with a detection at both ends
    saturation: float            # fraction of painted vertices clipped by the codec scale on any channel


def transition_pairs(fit: dict, smap: np.ndarray, n_out: int) -> np.ndarray:
    """``has_pair[k]`` for 30 fps frame ``k``: some track is detected at both ``smap[k]`` and ``smap[k+4]``."""
    detected = {}
    for f, t in zip(fit["frame_index"].tolist(), fit["track_id"].tolist()):
        detected.setdefault(int(t), set()).add(int(f))
    has = np.zeros(max(n_out - STRIDE, 0), dtype=bool)
    for k in range(len(has)):
        fa, fb = int(smap[k]), int(smap[k + STRIDE])
        has[k] = any(fa in fr and fb in fr for fr in detected.values())
    return has


def window_coverage(has_pair: np.ndarray, n_out: int) -> np.ndarray:
    """Number of transitions (of 8) with a hand pair for every admissible window start."""
    starts = n_out - (VIDEO_FRAMES - 1) * STRIDE
    if starts <= 0:
        return np.zeros(0, dtype=np.int64)
    cov = np.zeros(starts, dtype=np.int64)
    for j in range(TRANSITIONS):
        cov += has_pair[STRIDE * j: STRIDE * j + starts].astype(np.int64)
    return cov


def hand_window_meshes(
    fit: dict, *, start: int, smap: np.ndarray, width: int, height: int,
    intrinsics: np.ndarray | None, extrinsics: np.ndarray | None, world_to_cam: bool = True,
    compensate: bool = True, use_translation: bool = False, camera_conf: np.ndarray | None = None,
    camera_min_conf: float = 0.0,
) -> HandMeshes:
    frames_30 = [start + STRIDE * k for k in range(VIDEO_FRAMES)]
    src = [int(smap[k]) for k in frames_30]
    rec = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(fit["frame_index"], fit["track_id"]))}
    needed = sorted({i for (f, _), i in rec.items() if f in set(src)})
    verts = {}
    if needed:
        needed_arr = np.asarray(needed)
        for hand in (False, True):
            sel = needed_arr[fit["is_right"][needed_arr].astype(bool) == hand]
            if len(sel):
                v = mano_vertices(hand, fit["betas"][sel], fit["theta"][sel])
                for i, vi in zip(sel.tolist(), v):
                    verts[i] = vi

    if intrinsics is not None:
        fx, fy, cx0, cy0 = float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
    else:  # no calibration: a generic ego lens (~55 deg horizontal) keeps the depth estimate metric-ish
        fx = fy = 0.54 * width
        cx0, cy0 = width / 2.0, height / 2.0
    K_inv = np.linalg.inv(np.array([[fx, 0, cx0], [0, fy, cy0], [0, 0, 1]], dtype=np.float64))

    def root(i: int, depth: float | None = None) -> tuple[np.ndarray, float]:
        s, cx, cy = (float(x) for x in fit["cam"][i])
        z = fx / max(s, 1e-6) if depth is None else depth
        r = z * (K_inv @ np.array([cx, cy, 1.0]))
        return r.astype(np.float32), float(z)

    def project(x: np.ndarray) -> np.ndarray:
        z = np.maximum(x[:, 2], 1e-3)
        return np.stack((x[:, 0] / z * fx + cx0, x[:, 1] / z * fy + cy0), axis=-1)

    half = height / 2.0
    mesh_verts = np.zeros((VIDEO_FRAMES, MAX_HANDS, MANO_VERTS, 3), dtype=np.float32)
    mesh_colors = np.zeros((VIDEO_FRAMES, MAX_HANDS, MANO_VERTS, 3), dtype=np.float32)
    is_right = np.zeros((VIDEO_FRAMES, MAX_HANDS), dtype=bool)
    present = np.zeros((VIDEO_FRAMES, MAX_HANDS), dtype=bool)
    camera_delta = np.zeros((TRANSITIONS, 9), dtype=np.float32)
    camera_valid = np.zeros(TRANSITIONS, dtype=bool)
    pairs = np.zeros(TRANSITIONS, dtype=np.int64)
    painted = saturated = 0

    def ndc(i: int) -> np.ndarray:
        s, cx, cy = fit["cam"][i]
        v = verts[i]
        uv = v[:, :2] * s + np.array([cx, cy], np.float32)
        return np.stack([-(uv[:, 0] - width / 2.0) / half, -(uv[:, 1] - half) / half, 1.0 + v[:, 2]], axis=-1)

    def place(image: int, slot: int, i: int, colors: np.ndarray | None = None) -> None:
        mesh_verts[image, slot] = ndc(i)
        is_right[image, slot] = bool(fit["is_right"][i])
        present[image, slot] = True
        if colors is not None:
            mesh_colors[image, slot] = colors

    # Anchor: hand footprint at the window start (every detection at the source frame).
    for slot, i in enumerate([i for (f, _), i in rec.items() if f == src[0]][:MAX_HANDS]):
        place(0, slot, i)

    for j in range(TRANSITIONS):
        fa, fb = src[j], src[j + 1]
        R, t = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        if extrinsics is not None:
            ia, ib = min(fa, len(extrinsics) - 1), min(fb, len(extrinsics) - 1)
            R_cam, t_cam = relative_pose(extrinsics, ia, ib, world_to_cam)
            if not use_translation:
                t_cam = np.zeros(3, dtype=np.float32)
            camera_delta[j] = encode_camera_delta(torch.from_numpy(t_cam), torch.from_numpy(R_cam)).numpy()
            camera_valid[j] = True if camera_conf is None else bool(
                min(camera_conf[ia], camera_conf[ib]) >= camera_min_conf)
            if compensate:
                R, t = R_cam, t_cam
        slot = 0
        for (f, tid), i in rec.items():
            if f != fa or (fb, tid) not in rec or slot == MAX_HANDS:
                continue
            k = rec[(fb, tid)]
            ra, za = root(i)
            rb, _ = root(k, depth=za)
            xa = ra + verts[i]
            xb = (rb + verts[k]) @ R.T + t
            pa, pb = project(xa), project(xb)
            uvd = np.concatenate((pb - pa, (xb[:, 2:3] - xa[:, 2:3])), axis=-1).astype(np.float32)  # (du px, dv px, dd m)
            rgb, z = encode_uvd(uvd, np.ones(len(uvd), dtype=bool), width, SHRINK_D_M["kling"])
            place(j + 1, slot, i, rgb.astype(np.float32))
            slot += 1
            painted += len(uvd)
            saturated += int((np.abs(z) >= 1.0).any(axis=-1).sum())
        pairs[j] = slot

    return HandMeshes(verts=mesh_verts, colors=mesh_colors, is_right=is_right, present=present,
                      camera_delta=camera_delta, camera_valid=camera_valid, pairs_per_transition=pairs,
                      saturation=saturated / max(painted, 1))


@torch.no_grad()
def hand_pixels(verts: Tensor, colors: Tensor, is_right: Tensor, present: Tensor, height: int, width: int,
                device: torch.device | str = "cpu") -> dict:
    """Rasterise a batch of :class:`HandMeshes` (``[B, 9, MAX_HANDS, ...]`` CPU tensors) on ``device`` into the
    pixel fields of a track window: ``track_rgb`` uint8 ``[B, 9, H, W, 3]``, ``track_foreground`` /
    ``track_role_px`` ``[B, 9, H, W]`` (hands = body, role 1), ``track_delta`` float16 ``[B, 8, H, W, 3]``
    (normalised displacement decoded from the painted colour), ``head_mask`` ``[B, H, W]`` (hand footprint at
    the window start).  Meshes are assembled on the CPU and moved once, so the GPU path has no per-image syncs."""
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer.mesh.rasterizer import rasterize_meshes
    from pytorch3d.structures import Meshes

    b, n = present.shape[:2]
    track_rgb = torch.zeros(b, n, height, width, 3, dtype=torch.uint8, device=device)
    foreground = torch.zeros(b, n, height, width, dtype=torch.bool, device=device)
    delta = torch.zeros(b, n - 1, height, width, 3, dtype=torch.float16, device=device)
    present, is_right = present.cpu(), is_right.cpu()
    images = present.any(dim=-1).nonzero()
    if len(images):
        faces = {hand: torch.as_tensor(np.asarray(mano_model(hand).faces, dtype=np.int64)) for hand in (False, True)}
        vs, fs, fcs = [], [], []
        for i, j in images.tolist():
            slots = present[i, j].nonzero().flatten().tolist()
            f = torch.cat([faces[bool(is_right[i, j, k])] + MANO_VERTS * m for m, k in enumerate(slots)])
            vs.append(verts[i, j, slots].reshape(-1, 3).float())
            fs.append(f)
            fcs.append(colors[i, j, slots].reshape(-1, 3).float()[f])
        v_sizes, f_sizes = [len(v) for v in vs], [len(f) for f in fs]
        v_all = torch.cat(vs).to(device, non_blocking=True).split(v_sizes)
        f_all = torch.cat(fs).to(device, non_blocking=True).split(f_sizes)
        pix_to_face, _, bary, _ = rasterize_meshes(Meshes(verts=list(v_all), faces=list(f_all)), image_size=(height, width),
                                                   blur_radius=0.0, faces_per_pixel=1, bin_size=0)
        rgb = interpolate_face_attributes(pix_to_face, bary, torch.cat(fcs).to(device, non_blocking=True))[..., 0, :]
        hit = pix_to_face[..., 0] >= 0                                                     # [M, H, W]
        bi, ji = images[:, 0].to(device), images[:, 1].to(device)
        anchor = (ji == 0).view(-1, 1, 1, 1)
        painted = torch.where(anchor, torch.full_like(rgb, 128.0), rgb.round().clamp(0, 255))
        track_rgb[bi, ji] = torch.where(hit[..., None], painted, torch.zeros_like(painted)).to(torch.uint8)
        foreground[bi, ji] = hit
        moving = ji > 0
        z = rgb[moving] / 127.5 - 1.0
        decoded = torch.sign(z) * (torch.expm1(z.abs() * math.log1p(MU)) / MU)
        delta[bi[moving], ji[moving] - 1] = torch.where(hit[moving][..., None], decoded, torch.zeros_like(decoded)).to(torch.float16)
    return {"track_rgb": track_rgb, "track_foreground": foreground, "track_role_px": foreground.to(torch.uint8),
            "track_delta": delta, "head_mask": foreground[:, 0].clone()}


__all__ = ["HandMeshes", "MAX_HANDS", "hand_pixels", "hand_window_meshes", "mano_vertices", "relative_pose",
           "transition_pairs", "window_coverage"]
