#!/usr/bin/env python
"""KlingHumanEgo-2.5M-5000H -> 30 fps re-encode, shard-parallel, multi-machine.

Source: 17.2M mp4 clips packed in 5192 tar shards (indexed by a parquet with
video_path/offset/size).  Output mirrors that layout so we never create 17M
small files on Ceph:

  {out_root}/{tag}/videos_30fps/job-XXXXXXXXX/shard-XXXXXX.tar   processed clips (same member names)
  {out_root}/{tag}/index/job-XXXXXXXXX/shard-XXXXXX.parquet      per-shard index (new offset/size + meta)
  {out_root}/{tag}/locks/job-XXXXXXXXX__shard-XXXXXX/            mkdir-claim so several machines share the queue
  {out_root}/logs/{host}.log

`tag` = one pass over the source, selected by ego_mani_score.  Production runs two passes so the
high-manipulation subset lands first:  --tag hi --score-min 0.5   then   --tag lo --score-max 0.5

Per clip: `-vf fps=30` (frame dup/drop on timestamps, total duration preserved),
libx264 crf18 veryfast, keep original resolution (source is ~512x288, so no
480p resize), drop audio.  Clips that are already exactly 30/1 CFR are remuxed
with `-c:v copy`.

Sub-commands
  run     claim shards and process them (one driver per machine)
  status  progress summary
  merge   join all per-shard indexes with the source parquet -> final parquet
  clear-stale-locks  remove lock dirs without a finished index (only when no driver is running!)
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC_PARQUET = "/ytech_face_algo_ssd/lxq/captioner-data/20260526_13370_1_caption_v4_ego_local.parquet"
OUT_ROOT = "/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M"
TARGET_FPS = 30
FFMPEG_TIMEOUT = 900

INDEX_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("blobstore_key", pa.string()),
        ("member", pa.string()),
        ("video_path", pa.string()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("nb_frames", pa.int32()),
        ("duration", pa.float64()),
        ("src_fps", pa.string()),
        ("src_nb_frames", pa.int32()),
        ("src_duration", pa.float64()),
        ("mode", pa.string()),  # encode | copy | fail
        ("error", pa.string()),
    ]
)


# --------------------------------------------------------------------------- helpers
def shard_rel(video_path: str) -> str:
    """'/.../brush_.../job-000000000/shard-000000.tar' -> 'job-000000000/shard-000000.tar'"""
    p = Path(video_path)
    return f"{p.parent.name}/{p.name}"


def paths_for(out_root: Path, tag: str, rel: str):
    stem = rel[: -len(".tar")]
    base = out_root / tag
    return {
        "tar": base / "videos_30fps" / f"{stem}.tar",
        "index": base / "index" / f"{stem}.parquet",
        "lock": base / "locks" / stem.replace("/", "__"),
    }


def log(fh, msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def ffprobe(path: str) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe rc={out.returncode}: {out.stderr.strip()[:300]}")
    j = json.loads(out.stdout)
    if not j.get("streams"):
        raise RuntimeError("ffprobe: no video stream")
    s = j["streams"][0]
    dur = s.get("duration") or j.get("format", {}).get("duration")
    return {
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "r_frame_rate": s.get("r_frame_rate", ""),
        "avg_frame_rate": s.get("avg_frame_rate", ""),
        "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else -1,
        "duration": float(dur) if dur not in (None, "N/A") else -1.0,
    }


# --------------------------------------------------------------------------- worker
_W = {}


def _worker_init(preset: str, crf: int, prefix: str):
    # prefix is unique per driver process so one driver's cleanup can never touch another's scratch dirs
    d = Path("/dev/shm") / f"{prefix}_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    _W["dir"] = d
    _W["preset"] = preset
    _W["crf"] = crf


def _process_clip(task):
    """task = (row_idx, key, src_tar, offset, size) -> dict(index row, plus 'data')."""
    row_idx, key, src_tar, offset, size = task
    d = _W["dir"]
    d.mkdir(parents=True, exist_ok=True)
    fin, fout = d / "in.mp4", d / "out.mp4"
    row = {
        "row_idx": row_idx, "blobstore_key": key, "member": "", "width": 0, "height": 0,
        "nb_frames": -1, "duration": -1.0, "src_fps": "", "src_nb_frames": -1, "src_duration": -1.0,
        "mode": "fail", "error": "", "data": b"",
    }
    try:
        fd = os.open(src_tar, os.O_RDONLY)
        try:
            hdr = os.pread(fd, 512, offset - 512)
            data = os.pread(fd, size, offset)
        finally:
            os.close(fd)
        if len(data) != size:
            raise RuntimeError(f"short read {len(data)}/{size}")
        member = hdr[:100].rstrip(b"\0").decode("utf-8", "replace") or f"{key.split(':')[-1]}"
        row["member"] = member
        fin.write_bytes(data)

        src = ffprobe(str(fin))
        row.update(src_fps=src["r_frame_rate"], src_nb_frames=src["nb_frames"], src_duration=src["duration"])
        already_30 = src["r_frame_rate"] == f"{TARGET_FPS}/1" and src["avg_frame_rate"] == f"{TARGET_FPS}/1"
        if already_30:
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(fin),
                   "-c:v", "copy", "-an", "-movflags", "+faststart", str(fout)]
            mode = "copy"
        else:
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                   "-i", str(fin), "-vf", f"fps={TARGET_FPS}", "-c:v", "libx264", "-preset", _W["preset"],
                   "-crf", str(_W["crf"]), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                   "-threads", "1", str(fout)]
            mode = "encode"
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if r.returncode != 0 or not fout.exists() or fout.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg rc={r.returncode}: {r.stderr.strip()[:300]}")
        out = ffprobe(str(fout))
        row.update(width=out["width"], height=out["height"], nb_frames=out["nb_frames"],
                   duration=out["duration"], mode=mode, data=fout.read_bytes())
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:500]
        row["mode"] = "fail"
        row["data"] = b""
    finally:
        for f in (fin, fout):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
    return row


# --------------------------------------------------------------------------- shard processing
def load_shard_table(src_parquet: str, score_min=None, score_max=None):
    """Return (table, {src_tar: sorted row indices}) restricted to score_min <= score < score_max.

    NaN scores only survive when score_min is None (i.e. they fall into the 'rest' pass)."""
    t = pq.read_table(src_parquet, columns=["blobstore_key", "video_path", "offset", "size", "ego_mani_score"])
    vp = t["video_path"].combine_chunks().dictionary_encode()
    codes = vp.indices.to_numpy(zero_copy_only=False)
    shards = vp.dictionary.to_pylist()
    score = t["ego_mani_score"].to_numpy(zero_copy_only=False)
    keep = np.ones(len(codes), dtype=bool)
    if score_min is not None:
        keep &= score >= score_min
    if score_max is not None:
        keep &= ~(score >= score_max)
    sel = np.nonzero(keep)[0]
    order = sel[np.argsort(codes[sel], kind="stable")]
    bounds = np.searchsorted(codes[order], np.arange(len(shards) + 1))
    per_shard = {shards[i]: order[bounds[i]:bounds[i + 1]] for i in range(len(shards))}
    return t, per_shard


def process_shard(pool, t, rows_idx, src_tar, rel, out_root: Path, tag, logfh, chunksize, limit_clips=0):
    p = paths_for(out_root, tag, rel)
    p["tar"].parent.mkdir(parents=True, exist_ok=True)
    p["index"].parent.mkdir(parents=True, exist_ok=True)
    if limit_clips:
        rows_idx = rows_idx[:limit_clips]
    if len(rows_idx) == 0:  # nothing selected in this shard for this pass -> empty index marks it done
        pq.write_table(INDEX_SCHEMA.empty_table(), p["index"])
        log(logfh, f"DONE {rel}: 0 clips selected")
        return 0
    sub = t.take(pa.array(rows_idx))
    keys = sub["blobstore_key"].to_pylist()
    offs = sub["offset"].to_pylist()
    sizes = sub["size"].to_pylist()
    tasks = [(int(ri), k, src_tar, int(o), int(s)) for ri, k, o, s in zip(rows_idx, keys, offs, sizes)]

    tmp_tar = p["tar"].with_suffix(".tar.tmp")
    out_rows = []
    n_fail = n_copy = 0
    t0 = time.time()
    with tarfile.open(tmp_tar, "w", format=tarfile.GNU_FORMAT) as tf:
        for i, row in enumerate(pool.imap_unordered(_process_clip, tasks, chunksize=chunksize), 1):
            data = row.pop("data")
            if row["mode"] == "fail" or not data:
                n_fail += 1
                row.update(video_path="", offset=-1, size=0)
            else:
                if row["mode"] == "copy":
                    n_copy += 1
                info = tarfile.TarInfo(name=row["member"])
                info.size = len(data)
                info.mtime = 0
                hdr_off = tf.offset
                tf.addfile(info, io.BytesIO(data))
                # single 512B header for names < 100 chars; verify layout so offset is trustworthy
                pad = (512 - len(data) % 512) % 512
                if tf.offset != hdr_off + 512 + len(data) + pad:
                    raise RuntimeError(f"unexpected tar layout for member {row['member']}")
                row.update(video_path=str(p["tar"]), offset=hdr_off + 512, size=len(data))
            out_rows.append(row)
            if i % 500 == 0:
                el = time.time() - t0
                log(logfh, f"  {rel}: {i}/{len(tasks)} clips, {i / el:.1f} clip/s, fail={n_fail}")
    os.replace(tmp_tar, p["tar"])
    tbl = pa.Table.from_pylist(out_rows, schema=INDEX_SCHEMA)
    tmp_idx = p["index"].with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp_idx)
    os.replace(tmp_idx, p["index"])
    el = time.time() - t0
    log(logfh, f"DONE {rel}: {len(tasks)} clips in {el / 60:.1f} min ({len(tasks) / el:.1f} clip/s), "
               f"copy={n_copy} fail={n_fail}, tar={p['tar'].stat().st_size / 1e6:.0f} MB")
    return n_fail


def cmd_run(a):
    out_root = Path(a.out_root)
    for d in ("videos_30fps", "index", "locks"):
        (out_root / a.tag / d).mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    logfh = open(out_root / "logs" / f"{host}.log", "a")
    log(logfh, f"start host={host} tag={a.tag} score=[{a.score_min},{a.score_max}) workers={a.workers} "
               f"preset={a.preset} crf={a.crf} pid={os.getpid()}")

    t, per_shard = load_shard_table(a.parquet, a.score_min, a.score_max)
    shards = sorted(per_shard)
    if a.shard_slice:
        s = slice(*[int(x) if x else None for x in a.shard_slice.split(":")])
        shards = shards[s]
    n_sel = sum(len(v) for v in per_shard.values())
    log(logfh, f"loaded {t.num_rows} rows, selected {n_sel} clips, {len(shards)} shards in scope")

    ctx = mp.get_context("fork")
    shm_prefix = f"kling30_{os.getpid()}"
    pool = ctx.Pool(a.workers, initializer=_worker_init, initargs=(a.preset, a.crf, shm_prefix))
    # Several shards are streamed through the one pool concurrently (one thread each) so the
    # straggler tail of a shard (few long clips left) does not leave the workers idle.
    shard_iter = iter(shards)
    it_lock = threading.Lock()
    stats = {"done": 0, "skip": 0, "fail": 0}
    stop = threading.Event()

    def claim_loop():
        while not stop.is_set():
            with it_lock:
                if a.max_shards and stats["done"] >= a.max_shards:
                    return
                src_tar = next(shard_iter, None)
            if src_tar is None:
                return
            rel = shard_rel(src_tar)
            p = paths_for(out_root, a.tag, rel)
            if p["index"].exists():
                stats["skip"] += 1
                continue
            try:
                p["lock"].mkdir()
            except FileExistsError:
                stats["skip"] += 1
                continue
            (p["lock"] / f"{host}.{os.getpid()}").touch()
            log(logfh, f"CLAIM {rel} ({len(per_shard[src_tar])} clips)")
            try:
                stats["fail"] += process_shard(pool, t, per_shard[src_tar], src_tar, rel, out_root, a.tag, logfh,
                                               a.chunksize, a.limit_clips)
                stats["done"] += 1
            except Exception:  # noqa: BLE001
                log(logfh, f"ERROR {rel}:\n{traceback.format_exc()}")
                # release the claim so another driver (or a rerun) can retry
                shutil.rmtree(p["lock"], ignore_errors=True)
                if a.stop_on_error:
                    stop.set()
                    raise

    try:
        with ThreadPoolExecutor(a.concurrent_shards) as ex:
            futs = [ex.submit(claim_loop) for _ in range(a.concurrent_shards)]
            for f in futs:
                f.result()
    finally:
        pool.close()
        pool.join()
        for d in Path("/dev/shm").glob(f"{shm_prefix}_*"):
            shutil.rmtree(d, ignore_errors=True)
    log(logfh, f"exit host={host}: done={stats['done']} skipped={stats['skip']} clip_failures={stats['fail']}")


# --------------------------------------------------------------------------- status / merge / locks
def _tags(out_root: Path):
    return sorted(d.name for d in out_root.iterdir() if (d / "index").is_dir())


def _index_files(out_root: Path, tag: str):
    return sorted((out_root / tag / "index").glob("job-*/shard-*.parquet"))


def cmd_status(a):
    out_root = Path(a.out_root)
    for tag in _tags(out_root):
        idx = _index_files(out_root, tag)
        locks = list((out_root / tag / "locks").glob("*"))
        n_clips = n_fail = n_copy = 0
        bytes_out = 0
        mts = []
        for f in idx:
            tbl = pq.read_table(f, columns=["mode", "size"])
            if tbl.num_rows == 0:
                continue
            mts.append(f.stat().st_mtime)
            n_clips += tbl.num_rows
            modes = tbl["mode"].to_pylist()
            n_fail += modes.count("fail")
            n_copy += modes.count("copy")
            bytes_out += pc.sum(tbl["size"]).as_py() or 0
        n_total_shards = a.total_shards
        print(f"[{tag}] shards done: {len(idx)}/{n_total_shards}  in-progress: {len(locks) - len(idx)}")
        print(f"[{tag}] clips done: {n_clips}  copy={n_copy}  fail={n_fail}  out={bytes_out / 1e12:.3f} TB")
        if len(mts) > 1:
            mts.sort()
            span_h = (mts[-1] - mts[0]) / 3600
            rate = (len(mts) - 1) / max(span_h, 1e-9)
            print(f"[{tag}] {rate:.1f} shards/h, {n_clips / max(span_h, 1e-9) / 3600:.1f} clip/s over {span_h:.1f} h "
                  f"-> ETA {(n_total_shards - len(idx)) / rate:.1f} h")
    for lg in sorted((out_root / "logs").glob("*.log")):
        with open(lg) as fh:
            lines = fh.readlines()
        print(f"--- {lg.name}: {lines[-1].strip() if lines else ''}")


def cmd_merge(a):
    out_root = Path(a.out_root)
    tags = a.tags.split(",") if a.tags else _tags(out_root)
    idx = [f for tag in tags for f in _index_files(out_root, tag)]
    print(f"merging {len(idx)} shard indexes from tags {tags}")
    new = pa.concat_tables([pq.read_table(f) for f in idx])
    new = new.sort_by("row_idx")
    src = pq.read_table(a.parquet).drop_columns(["blobstore_key"])  # already in the shard index
    src = src.rename_columns([f"src_{c}" if c in ("video_path", "offset", "size") else c for c in src.column_names])
    src = src.append_column("row_idx", pa.array(np.arange(src.num_rows), pa.int64()))
    joined = new.join(src, keys="row_idx", join_type="inner").sort_by("row_idx")
    ok = joined.filter(pc.field("mode") != "fail")
    bad = joined.filter(pc.field("mode") == "fail")
    cols = ["blobstore_key", "caption", "ego_mani_score", "video_path", "offset", "size", "member",
            "width", "height", "nb_frames", "duration", "src_fps", "src_nb_frames", "src_duration", "mode",
            "src_video_path", "src_offset", "src_size", "camera_npz_archive", "camera_npz_member",
            "wilor_npz_archive", "wilor_npz_member", "row_idx"]
    ok = ok.select(cols)
    out = out_root / a.merge_name
    pq.write_table(ok, out, row_group_size=65536, compression="zstd")
    pq.write_table(bad.select(cols + ["error"]), out_root / "failed_clips.parquet")
    print(f"wrote {out}: {ok.num_rows} rows; failed {bad.num_rows} -> failed_clips.parquet; "
          f"src rows {src.num_rows} (missing {src.num_rows - joined.num_rows})")


def cmd_reset_failed(a):
    """Drop index/tar/lock of finished shards that contain failed clips so a later `run` redoes them."""
    import re

    out_root = Path(a.out_root)
    pat = re.compile(a.error_regex) if a.error_regex else None
    n_shards = n_clips = 0
    for tag in _tags(out_root):
        if a.tag and tag != a.tag:
            continue
        for f in _index_files(out_root, tag):
            tbl = pq.read_table(f, columns=["mode", "error"])
            if tbl.num_rows == 0:
                continue
            fails = [e for m, e in zip(tbl["mode"].to_pylist(), tbl["error"].to_pylist()) if m == "fail"]
            if not fails or (pat and not any(pat.search(e or "") for e in fails)):
                continue
            rel = f"{f.parent.name}/{f.stem}.tar"
            p = paths_for(out_root, tag, rel)
            print(f"reset [{tag}] {rel}: {len(fails)} failed clips")
            if not a.dry_run:
                p["index"].unlink()
                if p["tar"].exists():
                    p["tar"].unlink()
                shutil.rmtree(p["lock"], ignore_errors=True)
            n_shards += 1
            n_clips += len(fails)
    print(f"{'would reset' if a.dry_run else 'reset'} {n_shards} shards ({n_clips} failed clips)")


def cmd_clear_stale_locks(a):
    out_root = Path(a.out_root)
    n = 0
    for tag in _tags(out_root):
        for lk in (out_root / tag / "locks").glob("*"):
            rel = lk.name.replace("__", "/") + ".tar"
            p = paths_for(out_root, tag, rel)
            if not p["index"].exists():
                shutil.rmtree(lk)
                tmp = p["tar"].with_suffix(".tar.tmp")
                if tmp.exists():
                    tmp.unlink()
                n += 1
    print(f"removed {n} stale locks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=SRC_PARQUET)
    ap.add_argument("--out-root", default=OUT_ROOT)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--tag", default="all", help="output sub-dir; one tag per pass (e.g. hi / lo)")
    r.add_argument("--score-min", type=float, default=None, help="keep clips with ego_mani_score >= this")
    r.add_argument("--score-max", type=float, default=None, help="keep clips with score < this (NaN kept)")
    r.add_argument("--workers", type=int, default=max(4, os.cpu_count() - 8))
    r.add_argument("--preset", default="veryfast")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--chunksize", type=int, default=2)
    r.add_argument("--concurrent-shards", type=int, default=4, help="shards streamed through the pool at once")
    r.add_argument("--shard-slice", default="", help="python slice over sorted shard list, e.g. '0:10'")
    r.add_argument("--max-shards", type=int, default=0)
    r.add_argument("--limit-clips", type=int, default=0, help="DEBUG ONLY: first N clips per shard")
    r.add_argument("--stop-on-error", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sp.add_parser("status")
    s.add_argument("--total-shards", type=int, default=5192)
    s.set_defaults(fn=cmd_status)
    m = sp.add_parser("merge")
    m.add_argument("--merge-name", default="KlingHumanEgo20M_30fps.parquet")
    m.add_argument("--tags", default="", help="comma list of passes to merge, e.g. 'hi'; default all")
    m.set_defaults(fn=cmd_merge)
    rf = sp.add_parser("reset-failed")
    rf.add_argument("--tag", default="")
    rf.add_argument("--error-regex", default="", help="only reset shards whose failures match this")
    rf.add_argument("--dry-run", action="store_true")
    rf.set_defaults(fn=cmd_reset_failed)
    c = sp.add_parser("clear-stale-locks")
    c.set_defaults(fn=cmd_clear_stale_locks)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
