#!/usr/bin/env python
"""P1.3 - resolution of the Track4D representation pipeline (mu-law RGB -> h264 -> Wan VAE).

For each probe window (``windows.jsonl``) the eight stride-4 displacement frames are compared with the
metric ground truth ``delta_xyz_cam`` at three stages:

  codec      mu-law uint8 quantisation only (encode -> decode, no video, no VAE)
  video      the released Track mp4 frames decoded back to metres (codec + h264)
  vae        the mp4 frames through the frozen Wan2.2 VAE (encode -> decode) -> metres  [what the model sees]

Errors are metric EPE (m) on valid pixels, split by role (body / object) and by GT displacement magnitude
bins, so the table reads "a 5 mm object displacement survives the pipeline with x mm error".  This is the
lower bound for every Track prediction error and the resolution floor for coupling transitions in latent space.

    source scripts/probe/env.sh; CUDA_VISIBLE_DEVICES=1 $PY scripts/probe/vae_oracle.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import JANUS_ROOT, load_windows  # noqa: E402
from janusact4d_rt2imperfect_v1.data import decode_track_window, unpack_scene_mask, window_indices  # noqa: E402
from janusact4d_rt2imperfect_v1.encoder import FrozenTrackEncoder  # noqa: E402
from preprocess.track3d_codec import Track3DCodec  # noqa: E402
from training.observation import center_pad_video  # noqa: E402

OUT = Path(os.environ.get("PROBE_OUT", "/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes"))
BATCH = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
BINS = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 10.0])   # metres
W = 320


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(OUT / "windows.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "vae_oracle"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    cfg = yaml.safe_load((JANUS_ROOT / "config.yaml").read_text())
    head = json.loads(Path(cfg["depth_stats"]).read_text())["cameras"]["head_camera"]
    enc = FrozenTrackEncoder.from_pretrained(cfg["track_vae"], depth_min_m=head["min_mm"] * .001,
                                             depth_max_m=head["max_mm"] * .001, device=torch.device("cuda:0"))
    norm = json.loads((BATCH / "norm.json").read_text())
    scale = np.asarray(norm.get("scale_xyz_metres") or norm.get("scale_xyz") or norm["scale"], dtype=np.float32)
    codec = Track3DCodec(mu=31.0)
    windows = load_windows(a.windows)
    if a.limit:
        windows = windows[: a.limit]
    stages = ("codec", "video", "vae")
    roles = ("body", "object")
    acc = {s: {r: dict(se=np.zeros(len(BINS) - 1), ae=np.zeros(len(BINS) - 1), n=np.zeros(len(BINS) - 1)) for r in roles} for s in stages}
    per_window = []
    for i, w in enumerate(windows):
        key, start = w["key"], w["start"]
        task, variant, name = key.split("/")
        ep = BATCH / task / variant / name
        with h5py.File(ep / "source.hdf5", "r") as src:
            total = src["observation/head_camera/depth"].shape[0]
        ids = window_indices(start, total)
        rows = np.asarray(ids.track_rows)
        with h5py.File(ep / "track4d" / f"phase{ids.phase}.h5", "r") as th:
            gt = th["delta_xyz_cam"][rows].astype(np.float32)                     # [8, H, W, 3]
            valid = np.unpackbits(th["valid_bits"][rows], axis=-1, bitorder="little")[..., :W].astype(bool)
        with h5py.File(ep / "masks.h5", "r") as m:
            mb = np.unpackbits(m["head_camera/mask_bits"][list(ids.track_source)], axis=-1, bitorder="little")[..., :W].astype(bool)
        body, obj = mb[:, 0] & valid, mb[:, 1] & valid & ~mb[:, 0]
        rgb_video = decode_track_window(ep / "track4d" / f"phase{ids.phase}.mp4", ids.first_track_row)   # [8, H, W, 3]
        rgb_codec = codec.encode(gt, scale)
        anchor = np.full((1,) + rgb_video.shape[1:], 128, dtype=np.uint8)
        pixels = torch.from_numpy(np.concatenate((anchor, rgb_video), axis=0))[None]
        lat = enc.encode_pixels(center_pad_video(pixels, 256, 320))
        dec = enc.decode(lat)[0].float()                                             # [3, 9, 240, 320] in [-1, 1]
        rgb_vae = ((dec.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()[1:]
        est = dict(codec=codec.decode(rgb_codec, scale), video=codec.decode(rgb_video, scale), vae=codec.decode(rgb_vae, scale))
        mag = np.linalg.norm(gt, axis=-1)
        bin_idx = np.clip(np.digitize(mag, BINS) - 1, 0, len(BINS) - 2)
        rec = dict(key=key, start=start, kind=w["kind"])
        for s in stages:
            err = np.linalg.norm(est[s] - gt, axis=-1)
            for r, msk in (("body", body), ("object", obj)):
                if msk.sum() == 0:
                    continue
                e, b = err[msk], bin_idx[msk]
                np.add.at(acc[s][r]["se"], b, e ** 2)
                np.add.at(acc[s][r]["ae"], b, e)
                np.add.at(acc[s][r]["n"], b, 1)
                rec[f"{s}_{r}_epe_mm"] = float(e.mean() * 1000)
        per_window.append(rec)
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(windows)}]", flush=True)
    table = {}
    for s in stages:
        for r in roles:
            n = acc[s][r]["n"]
            table[f"{s}/{r}"] = dict(
                bins_m=BINS.tolist(), n=n.tolist(),
                epe_mm=(acc[s][r]["ae"] / np.maximum(n, 1) * 1000).round(3).tolist(),
                rmse_mm=(np.sqrt(acc[s][r]["se"] / np.maximum(n, 1)) * 1000).round(3).tolist(),
                overall_epe_mm=float(acc[s][r]["ae"].sum() / max(n.sum(), 1) * 1000))
    (out / "summary.json").write_text(json.dumps(dict(windows=len(per_window), scale_xyz_m=scale.tolist(), table=table), indent=1))
    (out / "per_window.jsonl").write_text("\n".join(json.dumps(r) for r in per_window))
    print(f"{len(per_window)} windows; scale {scale}")
    print(f"{'stage/role':14s} overall  " + "  ".join(f"[{BINS[i]*1000:.0f},{BINS[i+1]*1000:.0f})mm" for i in range(len(BINS) - 1)))
    for k, v in table.items():
        print(f"{k:14s} {v['overall_epe_mm']:6.2f}   " + "  ".join(f"{e:8.2f}" for e in v["epe_mm"]))


if __name__ == "__main__":
    main()
