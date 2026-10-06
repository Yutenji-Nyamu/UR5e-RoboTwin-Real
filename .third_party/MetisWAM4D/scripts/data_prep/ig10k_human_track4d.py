#!/usr/bin/env python
"""IG-10K human demos: dense Track4D = per-pixel 3D displacement in the CAMERA frame.

For every source frame t (four stride-4 phases together cover every frame) we pair it with
t+4, run RAFT forward+backward, and lift the correspondence through the DA3 metric depth:

    X_t(p)        = D_t(p)   * K^-1 [p, 1]
    X_{t+4}(p')   = D_{t+4}(p') * K^-1 [p', 1],   p' = p + flow(p)
    delta_xyz_cam = X_{t+4}(p') - X_t(p)

The ego camera is fixed, so E0 = E1 = I and no ego-motion is subtracted; the result is
directly the camera-frame displacement.  Schema, validity rule (in-bounds, valid depth both
ends, forward-backward error < 1 px, same-class footprint) and stride-4 phasing are taken
from JanusTrack4d/preprocess/robotwin_imperfect.py so the human and robot Track4D match.

Intrinsics: IG-10K has no calibration. K is taken from DA3's own camera head (same model that
produced the depth, so the two are self-consistent), estimated once per task dir as the median
over a few frames -- it is one physical ZED2i throughout.

Output {anno_root}/{subset}/{task}/track4d.h5, one group per episode:
    source_frame_index  int32 (P,)              P = n-4 pairs, all four phases merged
    forward_flow_px     int16 (P,H,W,2)  px/16
    delta_xyz_cam       int16 (P,H,W,3)  0.1 mm    (0 where invalid)
    valid_bits          uint8 packbits    (P,H,W/8)
    attrs: K (3x3 at 288p), stride=4, units

MUST run with /usr/bin/python3.10 (see ig10k_human_depth_masks.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_human_depth_masks as DM  # noqa: E402  (cut_episode, read_frames, da3_model, log)
from ig10k_anno_reader import IG10KAnno  # noqa: E402

RAFT_WEIGHTS = DM.FILES / "depth_study_models/raft_large.pth"
STRIDE = 4
FLOW_UPDATES = 24
FB_MAX_PX = 1.0
STATIC_FLOW_PX = 0.5  # |flow| below this over 4 frames -> pixel is static -> displacement written as 0
FLOW_Q = 16.0        # int16 flow in 1/16 px
DELTA_Q = 10000.0    # int16 delta in 0.1 mm
K_FRAMES = 8         # frames used to estimate K per task dir


# --------------------------------------------------------------------------- geometry (parity with robotwin_imperfect.py)
def grid(h, w):
    y, x = np.mgrid[:h, :w]
    return np.stack((x, y), -1).astype("f4")


def sample(value, uv, nearest=False):
    import cv2
    return cv2.remap(value.astype("f4"), uv[..., 0], uv[..., 1],
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def lift_pair(z0, z1, flow, K):
    """Camera-frame displacement for a fixed camera (E0 = E1 = I)."""
    h, w = z0.shape
    xy = grid(h, w)
    uv = xy + flow
    inside = (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    z1s = sample(z1, uv)
    z1_ok = sample((np.isfinite(z1) & (z1 > 0)).astype("f4"), uv) >= 0.99999
    Kinv = np.linalg.inv(K).T
    ray0 = np.concatenate((xy, np.ones((h, w, 1), "f4")), -1) @ Kinv
    ray1 = np.concatenate((uv, np.ones((h, w, 1), "f4")), -1) @ Kinv
    delta = (ray1 * z1s[..., None] - ray0 * z0[..., None]).astype("f4")
    valid = inside & z1_ok & np.isfinite(z0) & (z0 > 0) & np.isfinite(delta).all(-1)
    delta[~valid] = 0
    return delta, valid


def same_footprint(src_mask, tgt_mask, flow):
    """Source pixel is on the mask and its flow target lands fully on the same mask."""
    uv = grid(*flow.shape[:2]) + flow
    return src_mask & (sample(tgt_mask.astype("f4"), uv) >= 0.99999)


def same_instance(src_ids, tgt_ids, flow):
    """Source pixel carries an instance id and its flow target lands on the same id.

    Stricter than the class-union footprint: a flow vector that slides off the arm onto the shelf
    behind it used to pass (both are foreground) and lifted to a metre-scale bogus displacement.
    """
    uv = grid(*flow.shape[:2]) + flow
    tgt = sample(tgt_ids, uv, nearest=True)
    return (src_ids > 0) & (tgt == src_ids)


def instance_map(e) -> np.ndarray:
    """uint8 (T,H,W): shipped ids (or 0/1 objects) with the SAM arm, where present, as ARM_ID."""
    ids = e.ids.copy()
    if e.arm is not None:
        ids[e.arm] = DM.PROF.ARM_ID
    return ids


# --------------------------------------------------------------------------- models
_RAFT = {}


def raft():
    if "m" not in _RAFT:
        import torch
        from torchvision.models.optical_flow import raft_large
        m = raft_large(weights=None).cuda().eval()
        m.load_state_dict(torch.load(RAFT_WEIGHTS, map_location="cpu", weights_only=True))
        _RAFT["m"] = m
    return _RAFT["m"]


def raft_pairs(rgb: np.ndarray, src: np.ndarray, tgt: np.ndarray, batch: int = 8):
    """Forward and backward flow for the (src[i], tgt[i]) pairs -> two float32 (P,H,W,2)."""
    import torch
    import torch.nn.functional as F

    m = raft()
    n, h, w = rgb.shape[:3]
    fwd = np.zeros((len(src), h, w, 2), np.float32)
    bwd = np.zeros_like(fwd)
    ph, pw = (-h) % 8, (-w) % 8
    with torch.inference_mode():
        for s in range(0, len(src), batch):
            a = torch.from_numpy(rgb[src[s:s + batch]]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            b = torch.from_numpy(rgb[tgt[s:s + batch]]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            a = F.pad(a, (0, pw, 0, ph), mode="replicate")
            b = F.pad(b, (0, pw, 0, ph), mode="replicate")
            x = torch.cat([a, b])
            y = torch.cat([b, a])
            flows = m(x, y, num_flow_updates=FLOW_UPDATES)[-1][:, :, :h, :w].permute(0, 2, 3, 1)
            k = a.shape[0]
            fwd[s:s + k] = flows[:k].cpu().numpy()
            bwd[s:s + k] = flows[k:].cpu().numpy()
    return fwd, bwd


def estimate_K(rgb: np.ndarray) -> np.ndarray:
    """Median DA3-predicted intrinsics over a few frames, rescaled to the 288p frame size."""
    import torch

    model, _ = DM.da3_model()
    n, h, w = rgb.shape[:3]
    idx = np.linspace(0, n - 1, min(K_FRAMES, n)).round().astype(int)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.inference(list(rgb[idx]), process_res=DM.DA3_PROCESS_RES,
                               process_res_method=DM.DA3_RES_METHOD)
    K = np.asarray(pred.intrinsics, dtype=np.float64)  # at processed resolution
    ph, pw = pred.processed_images.shape[1:3]
    Km = np.median(K, axis=0)
    S = np.diag([w / pw, h / ph, 1.0])
    return (S @ Km).astype(np.float32)


# --------------------------------------------------------------------------- per task dir
def process_task_dir(anno: IG10KAnno, video_root: Path, sub: str, d: str, scratch: Path, logfh, shard=None):
    """shard=(k, n): only episodes k, k+n, ... -> track4d.shard{k}of{n}.h5 (see merge-shards)."""
    import h5py

    td_src = video_root / sub / d
    info = json.loads((td_src / "meta" / "info.json").read_text())
    ego = DM.ego_key(info)
    eps = DM.episodes_of(td_src, ego)
    if shard is not None:
        eps = eps[shard[0]::shard[1]]
    out = anno.task_dir(sub, d) / (f"track4d.shard{shard[0]}of{shard[1]}.h5" if shard else "track4d.h5")
    tmp = out.with_suffix(".h5.tmp")
    t0 = time.time()
    n_pairs = 0
    K = None
    with h5py.File(tmp, "w") as f:
        f.attrs.update(complete=False, stride=STRIDE, method="raft_large fwd+bwd, 24 updates",
                       flow_units=f"int16, px * {FLOW_Q:.0f}", delta_units=f"int16, m * {DELTA_Q:.0f} (0.1 mm)",
                       frame="camera (fixed ego ZED2i, E=I)", depth_source="DA3 da3nested-v11",
                       K_source="DA3 camera head, per-dir median", fb_max_px=FB_MAX_PX,
                       valid_rule="in-bounds & depth>0 both ends & FB<1px & same-instance footprint",
                       static_rule=f"valid & |flow| < {STATIC_FLOW_PX} px -> delta written as 0",
                       ego_key=ego)
        for ei, ep in enumerate(eps):
            name = f"ep_{ep['episode']:05d}"
            clip = scratch / f"{sub}_{d}_{name}.mp4"
            try:
                DM.cut_episode(ep, clip)
                rgb = DM.read_frames(clip)
            finally:
                clip.unlink(missing_ok=True)
            n, h, w = rgb.shape[:3]
            if K is None:
                K = estimate_K(rgb)
                f.attrs["K"] = K
                DM.log(f"  {sub}/{d}: K fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}", logfh)
            e = anno.load(sub, d, ep["episode"], with_depth=True)
            if e.depth.shape[0] != n:
                raise RuntimeError(f"{name}: depth {e.depth.shape[0]} frames vs rgb {n}")
            ids = instance_map(e)  # per-instance footprint for validity
            src = np.concatenate([np.arange(p, n - STRIDE, STRIDE) for p in range(STRIDE)])
            src.sort()
            tgt = src + STRIDE
            fwd, bwd = raft_pairs(rgb, src, tgt)
            P = len(src)
            g = f.create_group(name)
            g.attrs.update(frames=n, pairs=P, from_s=ep["from_s"], to_s=ep["to_s"])
            g.create_dataset("source_frame_index", data=src.astype(np.int32))
            ds_flow = g.create_dataset("forward_flow_px", shape=(P, h, w, 2), dtype="i2",
                                       compression="gzip", compression_opts=4, shuffle=True,
                                       chunks=(1, h, w, 2))
            ds_delta = g.create_dataset("delta_xyz_cam", shape=(P, h, w, 3), dtype="i2",
                                        compression="gzip", compression_opts=4, shuffle=True,
                                        chunks=(1, h, w, 3))
            ds_valid = g.create_dataset("valid_bits", shape=(P, h, (w + 7) // 8), dtype="u1",
                                        compression="gzip", compression_opts=4, chunks=(1, h, (w + 7) // 8))
            n_valid = 0
            for j, (t, u) in enumerate(zip(src, tgt)):
                fb = np.linalg.norm(fwd[j] + sample(bwd[j], grid(h, w) + fwd[j]), axis=-1)
                delta, geom = lift_pair(e.depth[t], e.depth[u], fwd[j], K)
                valid = geom & (fb < FB_MAX_PX) & same_instance(ids[t], ids[u], fwd[j])
                delta[~valid] = 0
                # A pixel that does not move in the image over 4 frames is static: the only thing
                # its lifted delta carries is DA3's frame-to-frame depth jitter (median 7 mm, p90
                # 30 mm on the robot scenes), which the mu-law codec paints as motion.
                delta[valid & (np.linalg.norm(fwd[j], axis=-1) < STATIC_FLOW_PX)] = 0
                ds_flow[j] = np.clip(np.rint(fwd[j] * FLOW_Q), -32768, 32767).astype(np.int16)
                ds_delta[j] = np.clip(np.rint(delta * DELTA_Q), -32768, 32767).astype(np.int16)
                ds_valid[j] = np.packbits(valid, axis=-1, bitorder="little")
                n_valid += int(valid.sum())
            g.attrs["valid_px_per_pair"] = n_valid / max(P, 1)
            n_pairs += P
            if (ei + 1) % 5 == 0 or ei + 1 == len(eps):
                el = time.time() - t0
                DM.log(f"  {sub}/{d}: {ei + 1}/{len(eps)} eps, {n_pairs} pairs, "
                       f"{n_pairs / max(el, 1e-9):.1f} pair/s, eta {el / (ei + 1) * (len(eps) - ei - 1) / 60:.1f} min", logfh)
        f.attrs["complete"] = True
    os.replace(tmp, out)
    mb = out.stat().st_size / 1e6
    DM.log(f"DONE {sub}/{d}: {len(eps)} eps, {n_pairs} pairs, {(time.time() - t0) / 60:.1f} min, "
           f"{mb:.0f} MB ({mb * 1e3 / max(n_pairs, 1):.0f} kB/pair)", logfh)
    return n_pairs


def cmd_run(a):
    anno = IG10KAnno(Path(a.anno_root), Path(a.video_root))
    A = Path(a.anno_root)
    (A / "_locks_track4d").mkdir(parents=True, exist_ok=True)
    (A / "_logs").mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    tag = f"{host}.gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.{os.getpid()}"
    logfh = open(A / "_logs" / f"track4d.{tag}.log", "a")
    scratch = Path(a.scratch) / tag
    scratch.mkdir(parents=True, exist_ok=True)
    DM.log(f"start track4d {tag}", logfh)
    if a.only_dir:
        sub, d = a.only_dir.split("/")
        k, n = (int(x) for x in a.ep_shard.split("/")) if a.ep_shard else (0, 1)
        try:
            process_task_dir(anno, Path(a.video_root), sub, d, scratch, logfh, shard=(k, n))
        except Exception:  # noqa: BLE001
            DM.log(f"ERROR shard {k}/{n} {sub}/{d}:\n{traceback.format_exc()}", logfh)
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return
    dirs = [(sub, d) for sub, d in DM.task_dirs(Path(a.video_root))
            if (A / sub / d / "depth_index.json").exists() and (A / sub / d / "masks.h5").exists()]
    if a.shuffle:
        np.random.default_rng(abs(hash(tag)) % 2 ** 32).shuffle(dirs)
    n_done = 0
    try:
        for sub, d in dirs:
            out = A / sub / d / "track4d.h5"
            lock = A / "_locks_track4d" / f"{sub}__{d}"
            if out.exists():
                continue
            try:
                lock.mkdir()
            except FileExistsError:
                continue
            (lock / tag).touch()
            try:
                process_task_dir(anno, Path(a.video_root), sub, d, scratch, logfh)
                n_done += 1
                shutil.rmtree(lock, ignore_errors=True)
            except Exception:  # noqa: BLE001
                DM.log(f"ERROR {sub}/{d}:\n{traceback.format_exc()}", logfh)
                (A / sub / d / "track4d.h5.tmp").unlink(missing_ok=True)
                shutil.rmtree(lock, ignore_errors=True)
                if a.stop_on_error:
                    raise
            if a.max_dirs and n_done >= a.max_dirs:
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    DM.log(f"exit track4d {tag}: dirs={n_done}", logfh)


def cmd_merge_shards(a):
    """track4d.shard*.h5 of one dir -> track4d.h5; every episode must appear exactly once.

    K is estimated per shard from its first episode (same fixed camera, DA3 camera head), so the
    shards' K differ by a few px; the merged file keeps shard 0's K and records the others.
    """
    import h5py

    sub, d = a.only_dir.split("/")
    base = Path(a.anno_root) / sub / d
    info = json.loads((Path(a.video_root) / sub / d / "meta" / "info.json").read_text())
    want = {f"ep_{e['episode']:05d}" for e in DM.episodes_of(Path(a.video_root) / sub / d, DM.ego_key(info))}
    shards = sorted(base.glob("track4d.shard*.h5"))
    tmp = base / "track4d.h5.tmp"
    seen, Ks = set(), []
    with h5py.File(tmp, "w") as out:
        for s in shards:
            with h5py.File(s) as f:
                if not out.attrs:
                    out.attrs.update(dict(f.attrs))
                Ks.append(np.asarray(f.attrs["K"]).tolist())
                for k in f:
                    if k in seen:
                        raise RuntimeError(f"{d}: {k} in more than one shard")
                    f.copy(k, out)
                    seen.add(k)
        out.attrs["complete"] = True
        out.attrs["merged_from_shards"] = len(shards)
        out.attrs["shard_K"] = json.dumps(Ks)
    if seen != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{d}: shards cover {len(seen)}/{len(want)} episodes; missing {sorted(want - seen)[:5]}")
    os.replace(tmp, base / "track4d.h5")
    for s in shards:
        s.unlink()
    print(f"merged {len(shards)} shards -> {base / 'track4d.h5'} ({len(seen)} episodes)")


def cmd_status(a):
    A = Path(a.anno_root)
    import h5py
    done = pairs = size = 0
    ready = 0
    dirs = DM.task_dirs(Path(a.video_root))
    for sub, d in dirs:
        p = A / sub / d
        if (p / "depth_index.json").exists() and (p / "masks.h5").exists():
            ready += 1
        t = p / "track4d.h5"
        if t.exists():
            done += 1
            size += t.stat().st_size
            with h5py.File(t) as f:
                pairs += sum(int(f[k].attrs["pairs"]) for k in f)
    locks = len(list((A / "_locks_track4d").glob("*"))) if (A / "_locks_track4d").exists() else 0
    print(f"track4d dirs {done}/{len(dirs)} (inputs ready for {ready})  pairs {pairs}  "
          f"{size / 1e9:.2f} GB  {size / 1e3 / max(pairs, 1):.0f} kB/pair  in-flight {locks}")
    for lg in sorted((A / "_logs").glob("track4d.*.log")):
        tail = lg.read_text().strip().splitlines()
        print(f"--- {lg.name[-40:]}: {tail[-1][:110] if tail else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(DM.PROF.PROFILES), default="human")
    ap.add_argument("--anno-root", default=None, help="default: the profile's root")
    ap.add_argument("--video-root", default=None, help="default: the profile's root")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--scratch", default="/dev/shm/ig10k_track4d")
    r.add_argument("--max-dirs", type=int, default=0)
    r.add_argument("--shuffle", action="store_true")
    r.add_argument("--stop-on-error", action="store_true")
    r.add_argument("--only-dir", default="", help="SUB/DIR: just this dir, no lock")
    r.add_argument("--ep-shard", default="", help="K/N with --only-dir -> track4d.shardKofN.h5")
    r.set_defaults(fn=cmd_run)
    ms = sp.add_parser("merge-shards")
    ms.add_argument("--only-dir", required=True)
    ms.set_defaults(fn=cmd_merge_shards)
    s = sp.add_parser("status")
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    DM.apply_profile(a.profile)
    a.anno_root = a.anno_root or DM.OUT_ROOT
    a.video_root = a.video_root or DM.SRC_ROOT
    a.fn(a)


if __name__ == "__main__":
    main()
