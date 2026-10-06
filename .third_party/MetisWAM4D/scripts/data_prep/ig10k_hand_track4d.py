#!/usr/bin/env python
"""IG-10K human (53 MANO dirs): dense HAND Track4D in the same metric camera frame as the object
Track4D, from the shipped WiLoR MANO parameters.

Per frame and hand the LeRobot parquet carries mano_global_orient(3), mano_hand_pose(45),
mano_betas(10), pred_cam_t_full(3), focal_length(2)=5000, pred_keypoints_3d(63).  Conventions
verified against pred_keypoints_3d (0.0 mm):
    right hand: smplx MANO_RIGHT, flat_hand_mean=True, params as stored
    left  hand: MANO_RIGHT with the SAME params, then mirror x (HaMeR ran the flipped image)

Geometry: HaMeR's camera (f=5000, principal point at 1280x720 centre) gives the correct 2D
footprint but its Z (~31 m) is a crop-scale artefact, so absolute depth comes from DA3:
    Z_anchor  = median over visible vertices of ( D_DA3(uv_v) - z_rel_v )
    Z_v       = Z_anchor + z_rel_v                      (MANO's metric relative depth)
    X_v       = (u_v - cx) / fx * Z_v,  Y_v = (v_v - cy) / fy * Z_v     (DA3 intrinsics K)
    delta     = X(t+4) - X(t) per vertex, rasterised with barycentric interpolation.
Hand and object Track4D therefore share one frame and one K.

Output {anno}/{subset}/{task}/hand_track4d.h5, group per episode (same pairing as track4d.h5):
    source_frame_index int32 (P,)   delta_xyz_cam int16 0.1mm (P,H,W,3)   valid_bits packbits (P,H,W/8)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_human_depth_masks as DM  # noqa: E402
from ig10k_anno_reader import IG10KAnno  # noqa: E402

MANO_RIGHT = "/m2v_intern_v3/danglingwei/model_zoos/mano_v1_2/models/MANO_RIGHT.pkl"
STRIDE = 4
DELTA_Q = 10000.0
SRC_W, SRC_H = 1280, 720  # HaMeR camera lives at the source resolution
_M = {}


def mano(device):
    import smplx
    if "m" not in _M:
        m = smplx.MANO(MANO_RIGHT, is_rhand=True, use_pca=False, flat_hand_mean=True, batch_size=1).to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        _M["m"] = m.eval()
    return _M["m"]


def load_hands(task_dir: Path, ego: str):
    """{episode: {'left'|'right': dict of arrays (T,...)}} plus per-frame validity."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(task_dir / "data" / "chunk-*" / "*.parquet")))
    t = pq.read_table(files[0]) if len(files) == 1 else pa.concat_tables([pq.read_table(f) for f in files])
    ep = np.asarray(t["episode_index"].to_pylist())
    out = {}
    for hand in ("left", "right"):
        cols = {f: np.stack(t[f"observation.hand.{hand}.{ego}.{f}"].to_pylist()).astype(np.float32)
                for f in ("mano_global_orient", "mano_hand_pose", "mano_betas", "pred_cam_t_full",
                          "focal_length", "pred_keypoints_3d")}
        valid = np.abs(cols["pred_keypoints_3d"]).sum(1) > 0
        for e in np.unique(ep):
            sel = ep == e
            out.setdefault(int(e), {})[hand] = {k: v[sel] for k, v in cols.items()} | {"valid": valid[sel]}
    return out


def hand_verts(m, h: dict, i: int, is_left: bool, device):
    """MANO vertices (778,3) in HaMeR camera orientation, wrist-centred, plus wrist offset t."""
    import torch
    with torch.no_grad():
        o = m(betas=torch.from_numpy(h["mano_betas"][i:i + 1]).to(device),
              global_orient=torch.from_numpy(h["mano_global_orient"][i:i + 1]).to(device),
              hand_pose=torch.from_numpy(h["mano_hand_pose"][i:i + 1]).to(device))
    v = o.vertices[0].cpu().numpy()
    if is_left:
        v = v * np.array([-1, 1, 1], np.float32)
    return v


HAMER_CROP = 256  # the stored focal_length (5000) is defined on HaMeR's 256px crop


def project_hamer(v: np.ndarray, cam_t: np.ndarray, f: float, scale: float):
    """HaMeR full-image perspective projection, then scaled to the 288p frame.

    pred_cam_t_full pairs with the *full-image* focal f * max(W,H) / 256 (= 25000 px at 1280x720),
    not with the stored per-crop 5000; using 5000 shrinks the hand ~16x (measured: 69 px footprint).
    """
    f_full = f / HAMER_CROP * max(SRC_W, SRC_H)
    p = v + cam_t
    u = f_full * p[:, 0] / p[:, 2] + SRC_W / 2
    w = f_full * p[:, 1] / p[:, 2] + SRC_H / 2
    return np.stack([u, w], 1) * scale


def lift_to_da3(uv: np.ndarray, z_rel: np.ndarray, depth: np.ndarray, K: np.ndarray):
    """Metric 3D in the DA3 camera: anchor hand depth to DA3 at the visible vertices."""
    h, w = depth.shape
    ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    d = depth[vi, ui]
    ok = inb & np.isfinite(d) & (d > 0)
    if ok.sum() < 20:
        return None
    z_anchor = float(np.median(d[ok] - z_rel[ok]))
    Z = z_anchor + z_rel
    X = (uv[:, 0] - K[0, 2]) / K[0, 0] * Z
    Y = (uv[:, 1] - K[1, 2]) / K[1, 1] * Z
    return np.stack([X, Y, Z], 1).astype(np.float32)


def render_pairs(m, hands: dict, depth: np.ndarray, K: np.ndarray, scale: float, device):
    """-> (src_idx int32 (P,), delta float32 (P,H,W,3), valid bool (P,H,W))"""
    import torch
    from pytorch3d.renderer.mesh.rasterizer import rasterize_meshes
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import interpolate_face_attributes

    n, h, w = depth.shape
    faces = torch.as_tensor(m.faces.astype(np.int64), device=device)
    # per frame per hand: (uv, X_da3) or None
    cache = {}
    for hand in ("left", "right"):
        hd = hands[hand]
        for i in range(n):
            if not hd["valid"][i]:
                cache[(i, hand)] = None
                continue
            v = hand_verts(m, hd, i, hand == "left", device)
            uv = project_hamer(v, hd["pred_cam_t_full"][i], float(hd["focal_length"][i][0]), scale)
            X = lift_to_da3(uv, v[:, 2] - v[0, 2] * 0 - np.median(v[:, 2]), depth[i], K)
            cache[(i, hand)] = None if X is None else (uv, X)
    src = np.concatenate([np.arange(p, n - STRIDE, STRIDE) for p in range(STRIDE)])
    src.sort()
    delta_out = np.zeros((len(src), h, w, 3), np.float32)
    valid_out = np.zeros((len(src), h, w), bool)
    half = h / 2.0
    for j, t in enumerate(src):
        vs, cols, fcs, off = [], [], [], 0
        for hand in ("left", "right"):
            a, b = cache.get((t, hand)), cache.get((t + STRIDE, hand))
            if a is None or b is None:
                continue
            uv, Xa = a
            Xb = b[1]
            ndc = np.stack([-(uv[:, 0] - w / 2) / half, -(uv[:, 1] - half) / half, Xa[:, 2]], 1)
            vs.append(ndc.astype(np.float32))
            cols.append((Xb - Xa).astype(np.float32))
            fcs.append(faces + off)
            off += 778
        if not vs:
            continue
        V = torch.from_numpy(np.concatenate(vs)).to(device)[None]
        F = torch.cat(fcs)[None]
        C = torch.from_numpy(np.concatenate(cols)).to(device)
        pix_to_face, _, bary, _ = rasterize_meshes(Meshes(verts=V, faces=F), image_size=(h, w),
                                                   blur_radius=0.0, faces_per_pixel=1, bin_size=0)
        attr = interpolate_face_attributes(pix_to_face, bary, C[F[0]])[0, :, :, 0]
        hit = (pix_to_face[0, :, :, 0] >= 0).cpu().numpy()
        d = attr.cpu().numpy()
        d[~hit] = 0
        delta_out[j] = d
        valid_out[j] = hit
    return src.astype(np.int32), delta_out, valid_out


def process_task_dir(anno: IG10KAnno, video_root: Path, sub: str, d: str, logfh, device):
    import h5py
    import torch
    td = video_root / sub / d
    info = json.loads((td / "meta" / "info.json").read_text())
    ego = DM.ego_key(info)
    ratio = anno.depth_index(sub, d)["episodes"]
    first = next(iter(ratio.values()))
    scale = first["width"] / SRC_W
    with h5py.File(anno.task_dir(sub, d) / "track4d.h5") as f:
        K = np.asarray(f.attrs["K"], np.float32)
    hands_all = load_hands(td, ego)
    m = mano(device)
    out = anno.task_dir(sub, d) / "hand_track4d.h5"
    tmp = out.with_suffix(".h5.tmp")
    t0 = time.time()
    n_pairs = 0
    with h5py.File(tmp, "w") as f:
        f.attrs.update(complete=False, stride=STRIDE, delta_units=f"int16, m * {DELTA_Q:.0f} (0.1 mm)",
                       frame="DA3 camera (same K as track4d.h5); hand depth anchored to DA3 at visible vertices",
                       mano="smplx MANO_RIGHT flat_hand_mean=True; left hand = same params, vertices mirrored in x",
                       source="IG-10K shipped WiLoR MANO params (ego view)", K=K, ego_key=ego)
        for name, e in ratio.items():
            ep = int(name.split("_")[1])
            if ep not in hands_all:
                continue
            depth = anno.load_depth(sub, d, ep)
            n = depth.shape[0]
            hd = hands_all[ep]
            if any(len(hd[hh]["valid"]) != n for hh in ("left", "right")):
                DM.log(f"  WARN {sub}/{d} {name}: MANO rows {len(hd['left']['valid'])} != frames {n}, skipping", logfh)
                continue
            src, delta, valid = render_pairs(m, hd, depth, K, scale, device)
            g = f.create_group(name)
            g.attrs.update(frames=n, pairs=len(src), valid_px_per_pair=float(valid.sum() / max(len(src), 1)))
            g.create_dataset("source_frame_index", data=src)
            g.create_dataset("delta_xyz_cam", data=np.clip(np.rint(delta * DELTA_Q), -32768, 32767).astype(np.int16),
                             compression="gzip", compression_opts=4, shuffle=True, chunks=(1, n and depth.shape[1], depth.shape[2], 3))
            g.create_dataset("valid_bits", data=np.packbits(valid, axis=-1, bitorder="little"),
                             compression="gzip", compression_opts=4, chunks=(1, depth.shape[1], (depth.shape[2] + 7) // 8))
            n_pairs += len(src)
        f.attrs["complete"] = True
    os.replace(tmp, out)
    DM.log(f"DONE {sub}/{d}: {n_pairs} pairs, {(time.time() - t0) / 60:.1f} min, {out.stat().st_size / 1e6:.0f} MB", logfh)
    return n_pairs


def cmd_demo(a):
    """Side-by-side RGB | object track | hand track for a few episodes -> mp4 + contact sheet."""
    import cv2
    import h5py
    import torch
    device = torch.device("cuda")
    anno = IG10KAnno()
    video_root = Path(DM.SRC_ROOT)
    D = Path(anno.anno_root) / "demo_hand_track4d"
    D.mkdir(exist_ok=True)
    scales = np.asarray([0.033, 0.020, 0.058], np.float32)  # display scale only (AgiBot codec)
    mu = 31.0

    def enc(delta):
        q = np.clip(delta / scales, -1, 1)
        e = np.sign(q) * np.log1p(mu * np.abs(q)) / np.log1p(mu)
        return np.rint((e + 1) * 127.5).clip(0, 255).astype(np.uint8)

    sheet = []
    for sub, d, ep in [("imitator_human_v1", "human_H1", 0), ("imitator_human_v1", "human_H14", 3),
                       ("imitator_human_v1", "human_H58", 5)]:
        td = video_root / sub / d
        info = json.loads((td / "meta" / "info.json").read_text())
        ego = DM.ego_key(info)
        e = anno.load(sub, d, ep)
        with h5py.File(anno.task_dir(sub, d) / "track4d.h5") as f:
            K = np.asarray(f.attrs["K"], np.float32)
            g = f[f"ep_{ep:05d}"]
            obj_src = g["source_frame_index"][:]
            obj_delta = g["delta_xyz_cam"][:].astype(np.float32) / 10000
            obj_valid = np.unpackbits(g["valid_bits"][:], axis=-1, count=e.depth.shape[2], bitorder="little").astype(bool)
        hd = load_hands(td, ego)[ep]
        src, delta, valid = render_pairs(mano(device), hd, e.depth, K, e.depth.shape[2] / SRC_W, device)
        eps = DM.episodes_of(td, ego)
        row = next(r for r in eps if r["episode"] == ep)
        clip = Path("/dev/shm/demo_rgb.mp4")
        DM.cut_episode(row, clip)
        rgb = DM.read_frames(clip)
        frames = []
        for j, t in enumerate(src):
            o = np.zeros_like(rgb[t])
            k = np.nonzero(obj_src == t)[0]
            if len(k):
                o[obj_valid[k[0]]] = enc(obj_delta[k[0]])[obj_valid[k[0]]]
            hnd = np.zeros_like(rgb[t])
            hnd[valid[j]] = enc(delta[j])[valid[j]]
            both = o.copy()
            both[valid[j]] = hnd[valid[j]]
            frames.append(np.concatenate([rgb[t], o, both], 1))
        frames = np.stack(frames)
        out = D / f"{d}_ep{ep}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "30", "-i", "pipe:",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
                       input=frames.tobytes(), capture_output=True, check=True)
        cov = valid.reshape(len(valid), -1).any(1).mean()
        mag = np.linalg.norm(delta[valid], axis=-1) if valid.any() else np.zeros(1)
        k = int(np.argmax(valid.reshape(len(valid), -1).sum(1)))
        sheet.append(frames[k])
        print(f"{d} ep{ep}: {len(src)} pairs, hand present in {cov * 100:.0f}% of pairs, hand px/pair "
              f"{valid.sum() / len(src):.0f}, |delta| median {np.median(mag) * 1000:.1f} mm p95 {np.quantile(mag, .95) * 1000:.1f} mm -> {out.name}",
              flush=True)
    cv2.imwrite(str(D / "contact_sheet.png"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR))
    print("written", D)


def cmd_run(a):
    import torch
    device = torch.device("cuda")
    anno = IG10KAnno()
    video_root = Path(DM.SRC_ROOT)
    A = Path(anno.anno_root)
    (A / "_locks_hand").mkdir(exist_ok=True)
    tag = f"{socket.gethostname().split('.')[0]}.gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.{os.getpid()}"
    logfh = open(A / "_logs" / f"hand_track4d.{tag}.log", "a")
    dirs = [(sub, p.name) for sub in ("imitator_human_v1",) for p in sorted((A / sub).glob("*"))
            if (p / "track4d.h5").exists() and DM.mano_columns(video_root / sub / p.name)]
    if a.shuffle:
        np.random.default_rng(abs(hash(tag)) % 2 ** 32).shuffle(dirs)
    n = 0
    for sub, d in dirs:
        out = A / sub / d / "hand_track4d.h5"
        lock = A / "_locks_hand" / f"{sub}__{d}"
        if out.exists():
            continue
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        try:
            process_task_dir(anno, video_root, sub, d, logfh, device)
            n += 1
        except Exception:  # noqa: BLE001
            DM.log(f"ERROR {sub}/{d}:\n{traceback.format_exc()}", logfh)
            (A / sub / d / "hand_track4d.h5.tmp").unlink(missing_ok=True)
        finally:
            shutil.rmtree(lock, ignore_errors=True)
        if a.max_dirs and n >= a.max_dirs:
            break
    DM.log(f"exit {tag}: dirs={n}", logfh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("demo").set_defaults(fn=cmd_demo)
    r = sp.add_parser("run")
    r.add_argument("--max-dirs", type=int, default=0)
    r.add_argument("--shuffle", action="store_true")
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
