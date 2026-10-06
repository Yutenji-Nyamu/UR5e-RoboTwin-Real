#!/usr/bin/env python
"""KlingHumanEgo-2.5M-5000H `hi`: dense hand Track4D from WiLoR joints via MANO IK + mesh rasterisation.

WiLoR's npz has no MANO pose, but its 21 `joints_3d` ARE MANO joints (16 + 5 fingertips) and
`betas` is there, so the pose is recoverable by IK.  Per clip, per hand track:

    1. IK    batched Adam on GPU: fit (global_orient, hand_pose) so that MANO's 21 OpenPose-ordered
             joints match joints_3d (wrist-centred), given betas.
    2. Mesh  smplx MANO forward -> 778 vertices per frame; same vertex index = same surface point,
             so V[t+4] - V[t] is the per-vertex camera-frame displacement.  Hand-root translation
             is kept in X/Y and zeroed in Z (WiLoR's Z is weak-perspective bbox jitter).
    3. Cam   weak-perspective (s, cx, cy) per frame by least squares between joints_2d and the
             wrist-centred joints_3d XY, so the mesh lands exactly where WiLoR saw the hand.
    4. Raster pytorch3d rasterize_meshes at 512x288; per-vertex delta interpolated with the
             barycentric weights of the covering face; both hands in one mesh so occlusion is
             resolved by depth.  Encoded with Track3DCodec(mu=31): 128 grey = zero, 0 black = invalid.

Artifacts
    fit    {out}/mano_hi/job-*/shard-*.tar   {key}.npz per clip: theta (N,48), betas, cam (N,3),
                                              frame_index, track_id, is_right, joint_rmse  (~30 kB)
           {out}/mano_hi/index/...parquet
    render (optional bake)  same layout as track_hi with the mesh renderer

MUST run with /usr/bin/python3.10 (smplx, pytorch3d).  MANO: model_zoos/mano_v1_2/models.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

MANO_DIR = Path("/m2v_intern_v3/danglingwei/model_zoos/mano_v1_2/models")
SRC_PARQUET = "/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M/KlingHumanEgo20M_30fps_hi.parquet"
OUT_ROOT = Path("/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M")
STRIDE = 4
OUT_FPS = 30
MU = 31.0
MIN_CONF = 0.3
IK_ITERS = int(os.environ.get("IK_ITERS", "30"))  # analytic init lands at ~0.5 mm; this is polish
# MANO(smplx) 16 joints: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb.
# Fingertips are mesh vertices (HaMeR/WiLoR ids). OpenPose order: wrist, thumb, index, middle, ring, pinky.
TIP_VIDS = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}
OPENPOSE_FROM_MANO = [0, 13, 14, 15, "thumb", 1, 2, 3, "index", 4, 5, 6, "middle",
                      10, 11, 12, "ring", 7, 8, 9, "pinky"]


# --------------------------------------------------------------------------- MANO
_M = {}


def mano(is_right: bool, device):
    import smplx
    import torch
    key = (bool(is_right), str(device))
    if key not in _M:
        m = smplx.MANO(str(MANO_DIR / ("MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl")),
                       is_rhand=is_right, use_pca=False, flat_hand_mean=True, batch_size=1).to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _M[key] = m
    return _M[key]


def mano_forward(m, betas, theta):
    """theta (N,48) -> verts (N,778,3), joints21 (N,21,3), both wrist-centred."""
    import torch
    n = theta.shape[0]
    out = m(betas=betas.expand(n, -1), global_orient=theta[:, :3], hand_pose=theta[:, 3:],
            return_verts=True)
    v, j = out.vertices, out.joints[:, :16]
    tips = torch.stack([v[:, TIP_VIDS[k]] for k in ("thumb", "index", "middle", "ring", "pinky")], 1)
    j21 = torch.stack([j[:, i] if isinstance(i, int) else tips[:, ["thumb", "index", "middle", "ring", "pinky"].index(i)]
                       for i in OPENPOSE_FROM_MANO], 1)
    wrist = j21[:, :1]
    return v - wrist, j21 - wrist


MANO_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
# child joint used to define each MANO joint's bone direction (fingertip vertices for the last link)
MANO_CHILD = {1: 2, 2: 3, 3: "index", 4: 5, 5: 6, 6: "middle", 7: 8, 8: 9, 9: "pinky",
              10: 11, 11: 12, 12: "ring", 13: 14, 14: 15, 15: "thumb"}
_LEAN = {}


def lean_model(m, betas):
    """What the IK needs, for per-detection betas (N,10): rest joints (N,16,3), the 5 fingertip rest
    vertices (N,5,3) and their blend weights (5,16)."""
    import torch
    with torch.no_grad():
        v_shaped = m.v_template + torch.einsum("bl,mkl->bmk", betas, m.shapedirs)  # (N,778,3)
        J = torch.einsum("bik,ji->bjk", v_shaped, m.J_regressor)  # (N,16,3)
        tips = torch.tensor([TIP_VIDS[k] for k in ("thumb", "index", "middle", "ring", "pinky")], device=betas.device)
        return {"J_rest": J, "v_tips_rest": v_shaped[:, tips], "W_tips": m.lbs_weights[tips]}


def lean_forward(lm, theta):
    """theta (N,48) -> joints21 wrist-centred (N,21,3) using only the kinematic chain + 5 tip verts."""
    import torch
    from smplx.lbs import batch_rodrigues
    n = theta.shape[0]
    R = batch_rodrigues(theta.reshape(-1, 3)).view(n, 16, 3, 3)
    J_rest = lm["J_rest"]  # (n,16,3)
    # forward kinematics
    G = [None] * 16
    Jg = [None] * 16
    for k in range(16):
        p = MANO_PARENTS[k]
        if p < 0:
            G[k] = R[:, k]
            Jg[k] = J_rest[:, k]
        else:
            G[k] = G[p] @ R[:, k]
            Jg[k] = Jg[p] + torch.einsum("nij,nj->ni", G[p], J_rest[:, k] - J_rest[:, p])
    Jg = torch.stack(Jg, 1)  # (n,16,3)
    Gs = torch.stack(G, 1)  # (n,16,3,3)
    # tips via LBS on 5 vertices: sum_k W_vk (G_k (v - J_k) + Jg_k)
    W = lm["W_tips"]  # (5,16)
    v0 = lm["v_tips_rest"]  # (n,5,3)
    rel = v0[:, :, None, :] - J_rest[:, None, :, :]  # (n,5,16,3)
    moved = torch.einsum("nkij,nvkj->nvki", Gs, rel) + Jg[:, None, :, :]  # (n,5,16,3)
    tips = (W[None, :, :, None] * moved).sum(2)  # (n,5,3)
    names = ["thumb", "index", "middle", "ring", "pinky"]
    j21 = torch.stack([Jg[:, i] if isinstance(i, int) else tips[:, names.index(i)] for i in OPENPOSE_FROM_MANO], 1)
    return j21 - j21[:, :1]


def _rot_between(a, b):
    """Batched rotation matrices taking unit vectors a to b (N,3) -> (N,3,3)."""
    import torch
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True)[..., None]
    s = v.norm(dim=-1, keepdim=True)[..., None]
    K = torch.zeros(a.shape[0], 3, 3, device=a.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
    I = torch.eye(3, device=a.device).expand_as(K)
    R = I + K + K @ K * ((1 - c) / (s ** 2 + 1e-12))
    return torch.where(s.expand_as(R) < 1e-6, I, R)


def _mat_to_aa(R):
    import torch
    cos = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1, 1)
    ang = torch.acos(cos)
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
    return axis * ang[:, None]


def analytic_init(lm, tgt21):
    """Closed-form pose from joint positions: Kabsch for the wrist, then per-joint bone alignment
    walking down each finger. Twist about each bone is unobservable and left at zero."""
    import torch
    n = tgt21.shape[0]
    J_rest = lm["J_rest"]
    # target joints in MANO-16 order (tips separately), all wrist-centred
    op_idx = {v: i for i, v in enumerate(OPENPOSE_FROM_MANO)}
    T16 = torch.stack([tgt21[:, op_idx[k]] for k in range(16)], 1)  # (n,16,3)
    Ttip = {name: tgt21[:, op_idx[name]] for name in ("thumb", "index", "middle", "ring", "pinky")}
    # global orient: Kabsch between rest and target knuckle bases (joints 1,4,7,10,13)
    base = [1, 4, 7, 10, 13]
    P = J_rest[:, base] - J_rest[:, :1]
    Q = T16[:, base] - T16[:, :1]
    H = P.transpose(1, 2) @ Q
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    Rg = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    G = [None] * 16
    G[0] = Rg
    Rloc = [None] * 16
    Rloc[0] = Rg
    for k in range(1, 16):
        p = MANO_PARENTS[k]
        child = MANO_CHILD[k]
        rest_dir = (lm["v_tips_rest"][:, ["thumb", "index", "middle", "ring", "pinky"].index(child)] if isinstance(child, str)
                    else J_rest[:, child]) - J_rest[:, k]
        tgt_dir = (Ttip[child] if isinstance(child, str) else T16[:, child]) - T16[:, k]
        # express target direction in the parent's frame, align rest bone to it
        d_par = torch.einsum("nji,nj->ni", G[p], tgt_dir)  # G[p]^T @ tgt_dir
        Rk = _rot_between(rest_dir, d_par)
        Rloc[k] = Rk
        G[k] = G[p] @ Rk
    return torch.cat([_mat_to_aa(Rloc[k]) for k in range(16)], 1)  # (n,48)


def fit_pose(m, betas, target21, device, iters=IK_ITERS):
    """Batched IK: analytic init + short Adam polish on the lean forward model. betas (N,10)."""
    import torch
    if betas.shape[0] == 1:
        betas = betas.expand(target21.shape[0], -1)
    lm = lean_model(m, betas)
    tgt = target21 - target21[:, :1]
    with torch.no_grad():
        theta0 = analytic_init(lm, tgt)
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.03)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        j = lean_forward(lm, theta)
        loss = ((j - tgt) ** 2).sum(-1).mean() + 1e-4 * ((theta[:, 3:] - theta0[:, 3:]) ** 2).mean()
        loss.backward()
        opt.step()
        if it == iters // 2:
            for g in opt.param_groups:
                g["lr"] = 0.01
    with torch.no_grad():
        j = lean_forward(lm, theta)
        rmse = ((j - tgt) ** 2).sum(-1).mean(-1).sqrt()
    return theta.detach(), rmse


def weak_persp(j2d: np.ndarray, j3d_centred: np.ndarray):
    """(s, cx, cy) with j2d ~= s * XY + c, least squares over 21 joints."""
    xy = j3d_centred[:, :2]
    A = np.concatenate([xy.reshape(-1, 1), np.tile(np.eye(2), (21, 1))], 1)  # (42, 3)
    b = j2d.reshape(-1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol.astype(np.float32)


# --------------------------------------------------------------------------- data
_TARS = {}


def npz_from_tar(archive, member):
    tf = _TARS.get(archive)
    if tf is None:
        if len(_TARS) > 8:
            for t in _TARS.values():
                t.close()
            _TARS.clear()
        tf = _TARS[archive] = tarfile.open(archive)
    return np.load(io.BytesIO(tf.extractfile(member).read()))


def detections(z):
    """Confident detections, one per (frame, track); track falls back to handedness."""
    conf = z["confidence"]
    ok = conf >= MIN_CONF
    tid = z["track_id"] if "track_id" in z.files else z["is_right"].astype(np.int32)
    best = {}
    for i in np.nonzero(ok)[0]:
        k = (int(z["frame_index"][i]), int(tid[i]))
        if k not in best or conf[i] > conf[best[k]]:
            best[k] = i
    idx = np.array(sorted(best.values()), dtype=np.int64)
    return idx, tid


def fit_clip(row: dict, device) -> dict:
    """IK for every detection of the clip -> sparse MANO record."""
    import torch
    z = npz_from_tar(row["wilor_npz_archive"], row["wilor_npz_member"])
    idx, tid = detections(z)
    n = len(idx)
    theta = np.zeros((n, 48), np.float32)
    cam = np.zeros((n, 3), np.float32)
    rmse = np.zeros(n, np.float32)
    betas = z["betas"][idx].astype(np.float32)
    j3 = z["joints_3d"][idx].astype(np.float32)
    j2 = z["joints_2d"][idx].astype(np.float32)
    isr = z["is_right"][idx].astype(bool)
    for hand in (False, True):
        sel = np.nonzero(isr == hand)[0]
        if not len(sel):
            continue
        m = mano(hand, device)
        b = torch.from_numpy(np.median(betas[sel], 0, keepdims=True)).to(device)  # one shape per hand
        tgt = torch.from_numpy(j3[sel]).to(device)
        th, r = fit_pose(m, b, tgt, device)
        theta[sel] = th.cpu().numpy()
        rmse[sel] = r.cpu().numpy()
        betas[sel] = b.cpu().numpy()
        for k, i in enumerate(sel):
            cam[i] = weak_persp(j2[i], j3[i] - j3[i][:1])
    return {"theta": theta, "betas": betas, "cam": cam, "frame_index": z["frame_index"][idx].astype(np.int32),
            "track_id": tid[idx].astype(np.int32), "is_right": isr, "joint_rmse_m": rmse,
            "root_xy_m": z["camera_translation"][idx, :2].astype(np.float32),
            "confidence": z["confidence"][idx].astype(np.float32)}


def fit_shard(rows: list, device, chunk: int = 60000) -> dict:
    """Load every clip's WiLoR detections, run ONE batched IK per hand over the whole shard,
    scatter back. Per-clip cost then is just I/O; the GPU sees ~1e5 detections at once."""
    import torch
    per = {}
    pool = {False: [], True: []}  # hand -> list of (clip_key, local_idx, j3, betas)
    for r in rows:
        key = r["blobstore_key"]
        try:
            z = npz_from_tar(r["wilor_npz_archive"], r["wilor_npz_member"])
            idx, tid = detections(z)
            isr = z["is_right"][idx].astype(bool)
            n = len(idx)
            rec = {"theta": np.zeros((n, 48), np.float32), "betas": np.zeros((n, 10), np.float32),
                   "cam": np.zeros((n, 3), np.float32), "joint_rmse_m": np.zeros(n, np.float32),
                   "frame_index": z["frame_index"][idx].astype(np.int32), "track_id": tid[idx].astype(np.int32),
                   "is_right": isr, "root_xy_m": z["camera_translation"][idx, :2].astype(np.float32),
                   "confidence": z["confidence"][idx].astype(np.float32)}
            j3 = z["joints_3d"][idx].astype(np.float32)
            j2 = z["joints_2d"][idx].astype(np.float32)
            for hand in (False, True):
                sel = np.nonzero(isr == hand)[0]
                if not len(sel):
                    continue
                b = np.median(z["betas"][idx][sel], 0).astype(np.float32)  # one shape per hand per clip
                rec["betas"][sel] = b
                for i in sel:
                    rec["cam"][i] = weak_persp(j2[i], j3[i] - j3[i][:1])
                pool[hand].append((key, sel, j3[sel], np.repeat(b[None], len(sel), 0)))
            per[key] = rec
        except Exception as e:  # noqa: BLE001
            per[key] = e
    for hand in (False, True):
        if not pool[hand]:
            continue
        m = mano(hand, device)
        tg = np.concatenate([p[2] for p in pool[hand]])
        be = np.concatenate([p[3] for p in pool[hand]])
        th_all = np.zeros((len(tg), 48), np.float32)
        r_all = np.zeros(len(tg), np.float32)
        for s in range(0, len(tg), chunk):
            th, rm = fit_pose(m, torch.from_numpy(be[s:s + chunk]).to(device),
                              torch.from_numpy(tg[s:s + chunk]).to(device), device)
            th_all[s:s + chunk] = th.cpu().numpy()
            r_all[s:s + chunk] = rm.cpu().numpy()
        off = 0
        for key, sel, j3, _ in pool[hand]:
            k = len(sel)
            per[key]["theta"][sel] = th_all[off:off + k]
            per[key]["joint_rmse_m"][sel] = r_all[off:off + k]
            off += k
    return per


# --------------------------------------------------------------------------- rendering
def encode(delta, scale):
    q = np.clip(np.asarray(delta, np.float32) / scale, -1.0, 1.0)
    e = np.sign(q) * np.log1p(MU * np.abs(q)) / np.log1p(MU)
    return np.rint((e + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def src_frame_map(n_out, src_fps, src_n):
    f = float(Fraction(src_fps))
    return np.minimum(np.rint(np.arange(n_out) * f / OUT_FPS).astype(int), max(src_n - 1, 0))


def render_clip(fit: dict, row: dict, scale: np.ndarray, device) -> np.ndarray:
    """-> uint8 (n_out, H, W, 3) track video frames."""
    import torch
    from pytorch3d.renderer.mesh.rasterizer import rasterize_meshes
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import interpolate_face_attributes

    n_out, h, w = int(row["nb_frames"]), int(row["height"]), int(row["width"])
    smap = src_frame_map(n_out, row["src_fps"], int(row["src_nb_frames"]))
    # per (frame, track) -> record index
    rec = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(fit["frame_index"], fit["track_id"]))}
    # MANO forward for every record (batched per hand)
    verts = np.zeros((len(fit["theta"]), 778, 3), np.float32)
    faces_by_hand = {}
    for hand in (False, True):
        sel = np.nonzero(fit["is_right"] == hand)[0]
        if not len(sel):
            continue
        m = mano(hand, device)
        faces_by_hand[hand] = torch.as_tensor(m.faces.astype(np.int64), device=device)
        with torch.no_grad():
            v, _ = mano_forward(m, torch.from_numpy(fit["betas"][sel[:1]]).to(device),
                                torch.from_numpy(fit["theta"][sel]).to(device))
        verts[sel] = v.cpu().numpy()
    frames = np.zeros((n_out, h, w, 3), np.uint8)
    half = h / 2.0
    for k in range(n_out - STRIDE):
        sa, sb = int(smap[k]), int(smap[k + STRIDE])
        vs, cols, fcs, off = [], [], [], 0
        for (f, t), i in rec.items():
            if f != sa or (sb, t) not in rec:
                continue
            j = rec[(sb, t)]
            va, vb = verts[i], verts[j]
            # camera-frame displacement: articulation in XYZ + root translation in XY only
            delta = (vb - va) + np.array([*(fit["root_xy_m"][j] - fit["root_xy_m"][i]), 0.0], np.float32)
            s, cx, cy = fit["cam"][i]
            uv = va[:, :2] * s + np.array([cx, cy], np.float32)
            ndc = np.stack([-(uv[:, 0] - w / 2) / half, -(uv[:, 1] - half) / half,
                            1.0 + va[:, 2] + (0.5 if fit["is_right"][i] else 0.0) * 0], -1)
            vs.append(ndc)
            cols.append(encode(delta, scale).astype(np.float32))
            fcs.append(faces_by_hand[bool(fit["is_right"][i])].cpu().numpy() + off)
            off += 778
        if not vs:
            continue
        V = torch.from_numpy(np.concatenate(vs)).to(device)[None]
        F = torch.from_numpy(np.concatenate(fcs)).to(device)[None]
        C = torch.from_numpy(np.concatenate(cols)).to(device)
        mesh = Meshes(verts=V, faces=F)
        pix_to_face, _, bary, _ = rasterize_meshes(mesh, image_size=(h, w), blur_radius=0.0,
                                                   faces_per_pixel=1, bin_size=0)
        face_cols = C[F[0]]  # (Fn, 3, 3)
        rgb = interpolate_face_attributes(pix_to_face, bary, face_cols)[0, :, :, 0]  # (h,w,3)
        hit = pix_to_face[0, :, :, 0] >= 0
        out = torch.zeros(h, w, 3, device=device)
        out[hit] = rgb[hit]
        frames[k] = out.round().clamp(0, 255).byte().cpu().numpy()
    return frames


def encode_video(frames: np.ndarray) -> bytes:
    n, h, w = frames.shape[:3]
    tmp = Path("/dev/shm") / f"mano_track_{os.getpid()}.mp4"
    try:
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(OUT_FPS), "-i", "pipe:", "-threads", "2",
             "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "10", "-preset", "veryfast", "-threads", "2",
             "-movflags", "+faststart", str(tmp)], input=frames.tobytes(), capture_output=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr.decode()[:200]}")
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- commands
def shard_rel(video_path):
    p = Path(video_path)
    return f"{p.parent.name}/{p.name}"


def cmd_check(a):
    """Fit + render one clip, write side-by-side PNGs and timing."""
    import cv2
    import pyarrow.parquet as pq
    import torch
    device = torch.device("cuda")
    t = pq.read_table(a.parquet).slice(a.row, 1).to_pylist()[0]
    scale = np.asarray(json.loads((OUT_ROOT / "track_hi" / "codec.json").read_text())["scale_xyz_metres"], np.float32)
    t0 = time.time()
    fit = fit_clip(t, device)
    t1 = time.time()
    frames = render_clip(fit, t, scale, device)
    t2 = time.time()
    print(f"detections {len(fit['theta'])}  joint RMSE median {np.median(fit['joint_rmse_m']) * 1000:.1f} mm "
          f"p90 {np.quantile(fit['joint_rmse_m'], .9) * 1000:.1f} mm | fit {t1 - t0:.1f}s render {t2 - t1:.1f}s "
          f"for {t['nb_frames']} frames")
    with open(t["video_path"], "rb") as fh:
        fh.seek(t["offset"])
        open("/tmp/rgb.mp4", "wb").write(fh.read(t["size"]))
    per = frames.reshape(len(frames), -1, 3).any(-1).mean(1)
    for tag, k in (("best", int(np.argmax(per))), ("mid", len(frames) // 2)):
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", "/tmp/rgb.mp4", "-vf", f"select=eq(n\\,{k})", "-vsync", "0",
                            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True)
        g = np.frombuffer(r.stdout, np.uint8).reshape(t["height"], t["width"], 3)
        cv2.imwrite(f"/tmp/mano_{tag}.png", cv2.cvtColor(np.concatenate([g, frames[k]], 1), cv2.COLOR_RGB2BGR))
        m = frames[k].any(-1)
        print(f"  {tag} frame {k}: hand px {m.sum()} ({100 * m.mean():.1f}%), rgb mean {frames[k][m].mean(0).round(0) if m.any() else '-'}")


def cmd_fit(a):
    """Sparse MANO fits for every clip, packed per video shard (GPU)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    device = torch.device("cuda")
    out_root = OUT_ROOT / "mano_hi"
    for d in ("index", "locks", "logs"):
        (out_root / d).mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    tag = f"{host}.gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.{os.getpid()}"
    logfh = open(out_root / "logs" / f"{tag}.log", "a")

    def log(m):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        logfh.write(line + "\n")
        logfh.flush()

    cols = ["blobstore_key", "video_path", "wilor_npz_archive", "wilor_npz_member"]
    t = pq.read_table(a.parquet, columns=cols)
    vp = t["video_path"].combine_chunks().dictionary_encode()
    codes = vp.indices.to_numpy(zero_copy_only=False)
    shards = vp.dictionary.to_pylist()
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(shards) + 1))
    sidx = np.arange(len(shards))
    if a.shuffle:
        np.random.default_rng(abs(hash(tag)) % 2 ** 32).shuffle(sidx)
    log(f"start {tag}: {len(shards)} shards, {t.num_rows} clips, IK iters {IK_ITERS}")
    n_done = 0
    for si in sidx:
        rel = shard_rel(shards[si])
        stem = rel[:-4]
        tar_out = out_root / f"{stem}.tar"
        idx_out = out_root / "index" / f"{stem}.parquet"
        lock = out_root / "locks" / stem.replace("/", "__")
        if idx_out.exists():
            continue
        try:
            lock.mkdir(parents=True)
        except FileExistsError:
            continue
        rows = t.take(pa.array(order[bounds[si]:bounds[si + 1]])).to_pylist()
        t0 = time.time()
        tar_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = tar_out.with_suffix(".tar.tmp")
        recs = []
        n_fail = 0
        try:
            fits = fit_shard(rows, device)  # one batched IK per hand for the whole shard
            with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tf:
                for r in rows:
                    rec = {"blobstore_key": r["blobstore_key"], "ok": True, "error": "", "n_det": 0,
                           "rmse_median_mm": 0.0, "mano_path": "", "offset": -1, "size": 0}
                    try:
                        fit = fits[r["blobstore_key"]]
                        if isinstance(fit, Exception):
                            raise fit
                        buf = io.BytesIO()
                        np.savez_compressed(buf, **fit)
                        data = buf.getvalue()
                        info = tarfile.TarInfo(name=f"{r['blobstore_key'].split(':')[-1]}.mano.npz")
                        info.size = len(data)
                        info.mtime = 0
                        hdr = tf.offset
                        tf.addfile(info, io.BytesIO(data))
                        rec.update(n_det=int(len(fit["theta"])),
                                   rmse_median_mm=float(np.median(fit["joint_rmse_m"]) * 1000) if len(fit["theta"]) else 0.0,
                                   mano_path=str(tar_out), offset=hdr + 512, size=len(data))
                    except Exception as e:  # noqa: BLE001
                        n_fail += 1
                        rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
                    recs.append(rec)
            os.replace(tmp, tar_out)
            tmp_idx = idx_out.with_suffix(".parquet.tmp")
            idx_out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(recs), tmp_idx)
            os.replace(tmp_idx, idx_out)
            shutil.rmtree(lock, ignore_errors=True)
            el = time.time() - t0
            log(f"DONE {rel}: {len(rows)} clips {el / 60:.1f} min ({len(rows) / el:.2f} clip/s) fail={n_fail} "
                f"tar={tar_out.stat().st_size / 1e6:.1f} MB")
            n_done += 1
        except Exception:  # noqa: BLE001
            log(f"ERROR {rel}:\n{traceback.format_exc()}")
            tmp.unlink(missing_ok=True)
            shutil.rmtree(lock, ignore_errors=True)
        if a.max_shards and n_done >= a.max_shards:
            break
    log(f"exit {tag}: shards={n_done}")


def cmd_status(a):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    out_root = OUT_ROOT / "mano_hi"
    idx = sorted((out_root / "index").glob("job-*/shard-*.parquet"))
    n = fail = 0
    size = 0
    rm = []
    for f in idx:
        t = pq.read_table(f, columns=["ok", "size", "rmse_median_mm"])
        n += t.num_rows
        fail += t.num_rows - pc.sum(t["ok"]).as_py()
        size += pc.sum(t["size"]).as_py() or 0
        rm += [x for x in t["rmse_median_mm"].to_pylist() if x]
    locks = len(list((out_root / "locks").glob("*"))) if (out_root / "locks").exists() else 0
    print(f"mano fits: shards {len(idx)}/5192 clips {n} fail {fail} out {size / 1e9:.2f} GB "
          f"({size / 1e3 / max(n, 1):.0f} kB/clip) in-flight {locks}  joint RMSE median {np.median(rm) if rm else 0:.1f} mm")
    for lg in sorted((out_root / "logs").glob("*.log")):
        tail = lg.read_text().strip().splitlines()
        print(f"--- {lg.name[-36:]}: {tail[-1][:110] if tail else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=SRC_PARQUET)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check")
    c.add_argument("--row", type=int, default=0)
    c.set_defaults(fn=cmd_check)
    f = sp.add_parser("fit")
    f.add_argument("--max-shards", type=int, default=0)
    f.add_argument("--shuffle", action="store_true")
    f.set_defaults(fn=cmd_fit)
    s = sp.add_parser("status")
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
