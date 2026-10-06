#!/usr/bin/env python
"""IG-10K human demonstrations -> 288p LeRobot copy (resolution pass only).

The source is already LeRobot v3 @ 30 fps CFR, so nothing temporal is touched: we only
rescale the RGB streams so they match the KlingHumanEgo-2.5M-5000H pretraining resolution.

  ego  (zed2i | zed)                1280x720 -> 512x288
  exo  (cam{1,2,3} | realsense{1,2,3} | rs{1,2,3})   640x480 -> 384x288

Short side is pinned to 288 and the aspect ratio is preserved, so no crop decision is
baked in -- the dataloader crops/pads to whatever the model wants.

Each LeRobot mp4 packs many episodes back to back and `meta/episodes/*.parquet` indexes
into it by timestamp, so the frame count and fps MUST survive the re-encode.  We pass no
`fps`/`-r` filter (1:1 frame mapping) and hard-fail any file whose frame count shifts.

Output mirrors the source tree under {out_root}/{subset}/{task_dir}/:
  videos/{rgb_key}/chunk-*/file-*.mp4   re-encoded, h264 crf18 yuv420p
  videos/{depth,mask_key}/...           symlinked to the source (not resized: interpolating
                                        depth codes or label colours would corrupt them)
  data/, meta/                          copied verbatim, except info.json image shapes are
                                        patched to the new resolution
  {out_root}/camera_aliases.json        per-dir {"ego": key, "exo": [keys]} so downstream
                                        code does not care about zed2i-vs-zed naming

Sub-commands: run (claim + encode, safe to run on several machines), status, verify.
`--profile robot` does the same for imitator_robot_v1 (see ig10k_profiles.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_profiles as PROF  # noqa: E402

SRC_ROOT = str(PROF.RAW_ROOT)
SUBSETS = PROF.PROFILES["human"]["subsets"]
INCLUDE = None  # set by --profile: restrict to these task dir names
TARGET_H = 288
EGO_HINT = "zed"  # ego camera keys all contain this; everything else RGB is exo


def rgb_keys(info: dict):
    ks = [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
    return [k for k in ks if not k.endswith(("_depth", "_mask"))]


def target_wh(shape) -> tuple[int, int]:
    _, h, w = shape
    return (round(w * TARGET_H / h / 2) * 2, TARGET_H)


def ffprobe_frames(path: Path) -> tuple[int, str, int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,r_frame_rate,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe rc={r.returncode}: {r.stderr.strip()[:200]}")
    s = json.loads(r.stdout)["streams"][0]
    return int(s["nb_read_packets"]), s["r_frame_rate"], int(s["width"]), int(s["height"])


def enumerate_units(src_root: Path, out_root: Path):
    """One unit = one mp4 to re-encode. Also returns per-dir bookkeeping."""
    units, dirs = [], []
    for sub in SUBSETS:
        for d in sorted(os.listdir(src_root / sub)):
            sd = src_root / sub / d
            if not sd.is_dir() or (INCLUDE is not None and d not in INCLUDE):
                continue
            info = json.loads((sd / "meta" / "info.json").read_text())
            keys = rgb_keys(info)
            ego = [k for k in keys if EGO_HINT in k]
            dirs.append({"sub": sub, "dir": d, "rgb": keys,
                         "ego": ego[0] if ego else None,
                         "exo": sorted(k for k in keys if EGO_HINT not in k),
                         "episodes": info["total_episodes"], "frames": info["total_frames"]})
            for k in keys:
                w, h = target_wh(info["features"][f"observation.images.{k}"]["shape"])
                for f in sorted((sd / "videos" / f"observation.images.{k}").glob("chunk-*/file-*.mp4")):
                    rel = f.relative_to(src_root)
                    units.append({"src": str(f), "dst": str(out_root / rel), "w": w, "h": h,
                                  "key": k, "is_ego": EGO_HINT in k,
                                  "lock": str(out_root / "_locks" / str(rel).replace("/", "__"))})
    # ego first so the stream we actually pretrain on lands before the exo views
    units.sort(key=lambda u: (not u["is_ego"], u["src"]))
    return units, dirs


def encode_one(u: dict, preset: str, crf: int) -> dict:
    src, dst = Path(u["src"]), Path(u["dst"])
    t0 = time.time()
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        n_src, fps_src, _, _ = ffprobe_frames(src)
        tmp = dst.with_name(dst.stem + ".tmp.mp4")  # keep .mp4 last so ffmpeg picks the muxer
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4",
             "-i", str(src), "-vf", f"scale={u['w']}:{u['h']}:flags=bicubic",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-an", "-movflags", "+faststart", "-threads", "4", str(tmp)],
            capture_output=True, text=True, timeout=7200,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg rc={r.returncode}: {r.stderr.strip()[:300]}")
        n_dst, fps_dst, w, h = ffprobe_frames(tmp)
        if n_dst != n_src or fps_dst != fps_src:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"frame/fps drift: {n_src}@{fps_src} -> {n_dst}@{fps_dst}")
        if (w, h) != (u["w"], u["h"]):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch: got {w}x{h} want {u['w']}x{u['h']}")
        os.replace(tmp, dst)
        return {"src": u["src"], "ok": True, "frames": n_src, "sec": time.time() - t0,
                "in_mb": src.stat().st_size / 1e6, "out_mb": dst.stat().st_size / 1e6, "error": ""}
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(u["lock"], ignore_errors=True)  # let a rerun retry this file
        return {"src": u["src"], "ok": False, "frames": 0, "sec": time.time() - t0,
                "in_mb": 0.0, "out_mb": 0.0, "error": f"{type(e).__name__}: {e}"[:400]}


def mirror_sidecars(src_root: Path, out_root: Path, dirs, logf):
    """Copy data/ + meta/ (with patched shapes) and symlink the depth/mask streams."""
    for d in dirs:
        sd, od = src_root / d["sub"] / d["dir"], out_root / d["sub"] / d["dir"]
        od.mkdir(parents=True, exist_ok=True)
        for name in ("data", "meta"):
            tgt = od / name
            if tgt.exists():
                shutil.rmtree(tgt)
            shutil.copytree(sd / name, tgt)
        info = json.loads((od / "meta" / "info.json").read_text())
        for k in d["rgb"]:
            feat = info["features"][f"observation.images.{k}"]
            w, h = target_wh(feat["shape"])
            feat["shape"] = [feat["shape"][0], h, w]
            feat.setdefault("info", {}).update({"video.height": h, "video.width": w})
        info["_metiswam4d_preprocess"] = {
            "source": str(sd), "target_short_side": TARGET_H,
            "note": "RGB streams rescaled (aspect preserved); fps/frame count unchanged; "
                    "depth and mask streams are symlinks to the untouched source",
        }
        (od / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        for vk in sorted((sd / "videos").iterdir()):
            if vk.name.split(".")[-1] in d["rgb"]:
                continue
            link = od / "videos" / vk.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(vk)
        print(f"  sidecars {d['sub']}/{d['dir']}", file=logf, flush=True)


def cmd_run(a):
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    (out_root / "_locks").mkdir(parents=True, exist_ok=True)
    (out_root / "_logs").mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    logf = open(out_root / "_logs" / f"{host}.log", "a")

    def log(m):
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", file=logf, flush=True)

    units, dirs = enumerate_units(src_root, out_root)
    log(f"host={host} {len(dirs)} task dirs, {len(units)} mp4 files, workers={a.workers}")
    (out_root / "camera_aliases.json").write_text(json.dumps(
        {f"{d['sub']}/{d['dir']}": {"ego": d["ego"], "exo": d["exo"]} for d in dirs}, indent=2))

    if not a.skip_sidecars:
        log("mirroring data/ meta/ and symlinking depth+mask streams")
        mirror_sidecars(src_root, out_root, dirs, logf)

    todo = []
    for u in units:
        if a.limit and len(todo) >= a.limit:
            break
        if Path(u["dst"]).exists():
            continue
        try:
            Path(u["lock"]).mkdir(parents=True)
        except FileExistsError:
            continue  # another machine owns this file
        todo.append(u)
    log(f"claimed {len(todo)} files")

    n_ok = n_bad = 0
    in_mb = out_mb = 0.0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(encode_one, u, a.preset, a.crf): u for u in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r["ok"]:
                n_ok += 1
                in_mb += r["in_mb"]
                out_mb += r["out_mb"]
            else:
                n_bad += 1
                log(f"FAIL {r['src']}: {r['error']}")
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"{i}/{len(todo)} files, ok={n_ok} fail={n_bad}, {el / 60:.1f} min, "
                    f"eta {el / i * (len(todo) - i) / 60:.1f} min, {in_mb / 1e3:.1f}->{out_mb / 1e3:.2f} GB")
    log(f"exit host={host}: ok={n_ok} fail={n_bad} in {(time.time() - t0) / 60:.1f} min")
    return 1 if n_bad else 0


def cmd_status(a):
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    units, dirs = enumerate_units(src_root, out_root)
    done = [u for u in units if Path(u["dst"]).exists()]
    locks = list((out_root / "_locks").glob("*")) if (out_root / "_locks").exists() else []
    ego_done = sum(1 for u in done if u["is_ego"])
    ego_all = sum(1 for u in units if u["is_ego"])
    in_mb = sum(Path(u["src"]).stat().st_size for u in done) / 1e6
    out_mb = sum(Path(u["dst"]).stat().st_size for u in done) / 1e6
    print(f"files {len(done)}/{len(units)} (ego {ego_done}/{ego_all})  in-flight {len(locks) - len(done)}")
    print(f"size {in_mb / 1e3:.1f} GB -> {out_mb / 1e3:.2f} GB  ({out_mb / max(in_mb, 1e-9) * 100:.1f}%)")
    print(f"task dirs {len(dirs)}, episodes {sum(d['episodes'] for d in dirs)}, "
          f"frames {sum(d['frames'] for d in dirs)} ({sum(d['frames'] for d in dirs) / 30 / 3600:.2f} h per stream)")
    for lg in sorted((out_root / "_logs").glob("*.log")) if (out_root / "_logs").exists() else []:
        tail = lg.read_text().strip().splitlines()
        print(f"--- {lg.name}: {tail[-1] if tail else ''}")


def cmd_verify(a):
    """Re-probe every output against its source: frame count, fps, resolution."""
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    units, dirs = enumerate_units(src_root, out_root)
    done = [u for u in units if Path(u["dst"]).exists()]
    print(f"verifying {len(done)} files")
    bad = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_verify_one, u): u for u in done}
        for i, fu in enumerate(as_completed(futs), 1):
            msg = fu.result()
            if msg:
                bad.append(msg)
                print("BAD", msg)
            if i % 50 == 0:
                print(f"  {i}/{len(done)}", flush=True)
    missing = [d for d in dirs if not (out_root / d["sub"] / d["dir"] / "meta" / "info.json").exists()]
    print(f"frame/fps/size mismatches: {len(bad)}; task dirs missing meta: {len(missing)}")
    return 1 if bad or missing else 0


def _verify_one(u):
    try:
        ns, fs, _, _ = ffprobe_frames(Path(u["src"]))
        nd, fd, w, h = ffprobe_frames(Path(u["dst"]))
        if (ns, fs) != (nd, fd) or (w, h) != (u["w"], u["h"]):
            return f"{u['dst']}: src {ns}@{fs} -> dst {nd}@{fd} {w}x{h} (want {u['w']}x{u['h']})"
    except Exception as e:  # noqa: BLE001
        return f"{u['dst']}: {type(e).__name__}: {e}"
    return ""


def main():
    global SUBSETS, INCLUDE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROF.PROFILES), default="human")
    ap.add_argument("--src-root", default=SRC_ROOT)
    ap.add_argument("--out-root", default=None, help="default: the profile's out_root")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--preset", default="veryfast")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--skip-sidecars", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sp.add_parser("status")
    s.set_defaults(fn=cmd_status)
    v = sp.add_parser("verify")
    v.add_argument("--workers", type=int, default=16)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    p = PROF.profile(a.profile)
    SUBSETS, INCLUDE = p["subsets"], p["include"]
    a.out_root = a.out_root or str(p["out_root"])
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
