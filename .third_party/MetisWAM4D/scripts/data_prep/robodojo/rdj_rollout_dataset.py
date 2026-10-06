"""Self-play data for RoboDojo: closed-loop rollouts of our own policy (recorded by ``metiswam4d/eval/rdj_deploy.py``
on official layouts >= 10, never the protocol layouts 0-9) -> RDJ_MetisWAM4D-compatible episodes.

For every recorded ``<record dir>/<variant>/layout_NNN.hdf5`` that passes the filter (success, or final process score
>= ``--min-score``) this writes ``<out>/<task>/rollout_<tag>/episodeN/{source.hdf5, track4d.h5, meta.json}``:

    source.hdf5   observation/{head,left,right}_camera/rgb (the recorded JPEG bytes, corpus byte order),
                  observation/head_camera/{depth (mm, robot pixels, f16), instance_id, intrinsic_cv, extrinsic_cv,
                  cam2world_gl}, joint_state/vector, metadata/instruction  -- the fields RDJEpisodeDataset reads
    track4d.h5    delta_uvd / role / eef20 / qpos / intrinsic_cv / extrinsic_cv, same schema as rdj_track4d.py
    meta.json     outcome, score, policy, source file

Geometry: SAPIEN robot-only depth + link ids from the recorded joint states and camera (the training renderer,
``official_geometry.DualX5Renderer``), per-link rigid motion from URDF FK (``rdj_track4d`` functions).

Sub-commands
    build     [GPU, SAPIEN] convert the recordings (multi-process, resumable)
    finalize  dataset root: symlinks to the original tasks / text cache / stats, index.jsonl (original train rows +
              rollout rows, ``--repeat`` copies per rollout row), episode_instructions_official.jsonl
    status    counts

    PYTHONPATH=.:<JanusTrack4d_260824> VK_ICD_FILENAMES=<sapien icd> CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.10 \
        scripts/data_prep/robodojo/rdj_rollout_dataset.py build --record <dir> --out <root> --tag v1 --workers 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json")

from metiswam4d.data.rt2.episode_dataset import SOURCE_FRAMES  # noqa: E402
from metiswam4d.data.rt2.text_cache import format_prompt  # noqa: E402
from metiswam4d.eval.rdj_observation import URDF, pose_wxyz_to_arm10  # noqa: E402

ORIGINAL = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D")
STRIDE = 4
SCHEMA = "metiswam4d.rdj_rollout_track4d.uvd.v1"


def _track_module():
    spec = importlib.util.spec_from_file_location("rdj_track4d", Path(__file__).with_name("rdj_track4d.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gl_to_cv(cam2world_gl: np.ndarray) -> np.ndarray:
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return (flip @ np.linalg.inv(np.asarray(cam2world_gl, np.float64)))


def episode_dir(out: Path, task: str, tag: str, episode: int) -> Path:
    return out / task / f"rollout_{tag}" / f"episode{episode}"


def list_recordings(record: Path) -> list[Path]:
    return sorted(record.glob("*/layout_*.hdf5"))


def accept(attrs, min_score: float, successes_only: bool) -> bool:
    if int(attrs["frames"]) < SOURCE_FRAMES:
        return False
    if bool(attrs["success"]):
        return True
    return (not successes_only) and float(attrs.get("final_score", 0.0)) >= min_score


def build_one(args) -> dict:
    path, out, tag, episode = args
    path = Path(path)
    try:
        return _build(path, Path(out), tag, int(episode))
    except Exception as exc:  # noqa: BLE001 - recorded, the shard goes on
        return {"source": str(path), "status": "error", "error": repr(exc)[:300], "trace": traceback.format_exc()[-1500:]}


_RENDERER = {}


def _renderer(intrinsic: np.ndarray, shape: tuple[int, int]):
    from preprocess.robodojo.official_geometry import DualX5Renderer
    key = (tuple(np.round(intrinsic, 4).ravel()), shape)
    if key not in _RENDERER:
        _RENDERER.clear()
        _RENDERER[key] = DualX5Renderer(URDF, {"head_camera": (intrinsic, shape + (3,))})
    return _RENDERER[key]


def _build(path: Path, out: Path, tag: str, episode: int) -> dict:
    T = _track_module()
    with h5py.File(path) as rec:
        attrs = dict(rec.attrs)
        task = str(attrs["task"])
        dest = episode_dir(out, task, tag, episode)
        if (dest / "meta.json").exists():
            previous = json.loads((dest / "meta.json").read_text())
            if Path(previous["source"]).resolve() != path.resolve():
                raise ValueError(f"episode collision at {dest}: {previous['source']} vs {path}")
            return {"source": str(path), "status": "exists", "task": task, "episode": episode}
        dest.mkdir(parents=True, exist_ok=True)
        started = time.time()
        n = int(attrs["frames"])
        qpos = rec["joint_state/vector"][:].astype(np.float64)                  # [n, 14]
        ee = rec["state/ee_poses_wxyz"][:].astype(np.float64)                   # [n, 14]
        score = rec["score"][:].astype(np.float32)
        head = rec["observation/head_camera"]
        k_live = head["intrinsic_live"][:].astype(np.float64)
        c2w = head["cam2world_gl"][:].astype(np.float64)
        live_shape = tuple(int(v) for v in head["live_shape"][:])
        jpegs = {cam: [bytes(rec[f"observation/{cam}/rgb"][t]) for t in range(n)]
                 for cam in ("head_camera", "left_camera", "right_camera")}
    renderer = _renderer(k_live, live_shape)
    K = renderer.intrinsics["head_camera"].astype(np.float64)                    # 320x240
    E = gl_to_cv(c2w)
    h, w = 240, 320
    id_to_link = {i + 1: name for i, name in enumerate(renderer.link_names)}
    joints = T.urdf_joints(URDF)
    states = {"left_arm": qpos[:, :6], "left_gripper": qpos[:, 6], "right_arm": qpos[:, 7:13], "right_gripper": qpos[:, 13]}
    poses = T.link_poses(states, joints)
    eef20 = np.stack([np.concatenate((pose_wxyz_to_arm10(ee[t, :7], qpos[t, 6]),
                                      pose_wxyz_to_arm10(ee[t, 7:14], qpos[t, 13])))
                      for t in range(n)]).astype(np.float32)                     # [n, 20] world, link6 + gripper
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    rays = np.stack(((xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)), -1)

    depth_all = np.zeros((n, h, w), np.float16)
    ids_all = np.zeros((n, h, w), np.uint32)
    for t in range(n):
        renderer.set_state({"left": qpos[t, :6], "right": qpos[t, 7:13]}, {"left": float(qpos[t, 6]), "right": float(qpos[t, 13])})
        rendered, _ = renderer.render({"head_camera": c2w.astype(np.float32)})
        links, depth_m = rendered["head_camera"]
        depth_all[t] = (depth_m * 1000.0).astype(np.float16)
        ids_all[t] = links.astype(np.uint32)

    tmp_src = dest / "source.tmp.hdf5"
    with h5py.File(tmp_src, "w") as src:
        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        for cam in ("head_camera", "left_camera", "right_camera"):
            g = src.create_group(f"observation/{cam}")
            d = g.create_dataset("rgb", (n,), dtype=vlen)
            for t in range(n):
                d[t] = np.frombuffer(jpegs[cam][t], dtype=np.uint8)
        g = src["observation/head_camera"]
        g.create_dataset("depth", data=depth_all, chunks=(1, h, w), compression="lzf")
        g.create_dataset("instance_id", data=ids_all, chunks=(1, h, w), compression="lzf")
        g.create_dataset("intrinsic_cv", data=np.broadcast_to(K.astype(np.float32), (n, 3, 3)))
        g.create_dataset("extrinsic_cv", data=np.broadcast_to(E[:3].astype(np.float32), (n, 3, 4)))
        g.create_dataset("cam2world_gl", data=np.broadcast_to(c2w.astype(np.float32), (n, 4, 4)))
        src.create_dataset("joint_state/vector", data=qpos.astype(np.float32))
        for side, sl, gi in (("left", slice(0, 6), 6), ("right", slice(7, 13), 13)):
            src.create_dataset(f"joint_state/{side}_arm", data=qpos[:, sl].astype(np.float32))
            src.create_dataset(f"joint_state/{side}_gripper", data=qpos[:, gi].astype(np.float32))
        src.create_dataset("score", data=score)
        m = src.create_group("metadata")
        m.create_dataset("instruction", data=str(attrs.get("instruction", "")), dtype=h5py.string_dtype("utf-8"))
        m.create_dataset("source_episode", data=str(path), dtype=h5py.string_dtype("utf-8"))
        m.create_dataset("geometry_scope", data="robot_and_gripper_only", dtype=h5py.string_dtype("utf-8"))
        m.create_dataset("schema_version", data=SCHEMA, dtype=h5py.string_dtype("utf-8"))
        m.create_dataset("source_frequency", data=25)
    os.replace(tmp_src, dest / "source.hdf5")

    tmp = dest / "track4d.tmp.h5"
    robot_pixels = 0
    with h5py.File(tmp, "w") as t4d:
        t4d.attrs.update(schema=SCHEMA, stride=STRIDE, camera="head_camera", coordinate_frame="source_head_camera_cv",
                         units="du px, dv px, dd m", width=w, height=h, frames=n, role_labels="0 unknown, 1 robot",
                         source=str(path), urdf=str(URDF),
                         eef20_layout="left xyz(3) rot6d(6) gripper(1) | right xyz(3) rot6d(6) gripper(1); world frame, link6",
                         quaternion_order="wxyz (simulator state/*_ee_pose)")
        t4d.create_dataset("intrinsic_cv", data=K.astype(np.float32))
        t4d.create_dataset("extrinsic_cv", data=E[:3].astype(np.float32))
        t4d.create_dataset("eef20", data=eef20)
        t4d.create_dataset("qpos", data=qpos.astype(np.float32))
        d_uvd = t4d.create_dataset("delta_uvd", (n - STRIDE, h, w, 3), dtype="f2", chunks=(1, h, w, 3),
                                   compression="lzf", shuffle=True)
        d_role = t4d.create_dataset("role", (n, h, w), dtype="u1", chunks=(1, h, w), compression="lzf")
        for t in range(n):
            depth = depth_all[t].astype(np.float64) * 1e-3
            ids = ids_all[t]
            valid = (ids > 0) & (depth > 0)
            d_role[t] = valid.astype(np.uint8)
            robot_pixels += int(valid.sum())
            if t >= n - STRIDE:
                continue
            pts = rays * depth[..., None]
            delta = T.rigid_delta(pts, ids, valid, id_to_link, poses, E, t, t + STRIDE)
            uv0, uv1 = T.project(pts, K), T.project(pts + delta, K)
            uvd = np.concatenate((uv1 - uv0, delta[..., 2:3]), -1)
            uvd[~valid] = 0.0
            d_uvd[t] = uvd.astype(np.float16)
        t4d.attrs["complete"] = True
    os.replace(tmp, dest / "track4d.h5")
    meta = {"task": task, "episode": episode, "frames": n, "status": "ok", "success": bool(attrs["success"]),
            "final_score": float(attrs.get("final_score", 0.0)), "success_steps": int(attrs.get("success_steps", 0)),
            "layout_id": int(attrs["layout_id"]), "variant": str(attrs["variant"]), "policy": str(attrs.get("policy", "{}")),
            "instruction": str(attrs.get("instruction", "")), "source": str(path),
            "robot_pixels_per_frame": robot_pixels / n, "seconds": round(time.time() - started, 1)}
    (dest / "meta.json").write_text(json.dumps(meta))
    return {"source": str(path), "status": "ok", "task": task, "episode": episode, "frames": n, "seconds": meta["seconds"]}


def cmd_build(args) -> None:
    out, record = Path(args.out), Path(args.record)
    jobs, skipped = [], 0
    counter: dict[str, int] = {}
    tasks = set(args.tasks.split(",")) if args.tasks else None
    for path in list_recordings(record):
        if tasks is not None and path.parent.name.split("@")[0].removesuffix("_random") not in tasks:
            skipped += 1
            continue
        try:
            with h5py.File(path) as rec:
                attrs = dict(rec.attrs)
        except OSError as exc:
            log = out / "_logs"
            log.mkdir(parents=True, exist_ok=True)
            with (log / "bad_recordings.jsonl").open("a") as handle:
                handle.write(json.dumps({"source": str(path), "error": repr(exc)}) + "\n")
            skipped += 1
            continue
        task_name = str(attrs["task"])
        if tasks is not None and task_name.removesuffix("_random") not in tasks:
            skipped += 1
            continue
        if not accept(attrs, args.min_score, args.successes_only):
            skipped += 1
            continue
        task = str(attrs["task"])
        # deterministic episode number: <layout id> + 1000 * <ordinal of the recording variant for this task>
        variant = str(attrs["variant"])
        key = f"{task}:{variant}"
        if key not in counter:
            counter[key] = len([k for k in counter if k.startswith(task + ":")])
        episode = args.episode_offset + int(attrs["layout_id"]) + 1000 * counter[key]
        jobs.append((str(path), str(out), args.tag, episode))
    print(json.dumps({"recordings": len(jobs) + skipped, "accepted": len(jobs), "skipped": skipped}), flush=True)
    log = out / "_logs"
    log.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, open(log / f"build_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "a") as handle:
        for i, result in enumerate(pool.imap_unordered(build_one, jobs)):
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if result["status"] != "exists":
                print(f"[{i + 1}/{len(jobs)}] {result['status']} {result.get('task')} ep{result.get('episode')} "
                      f"{result.get('seconds', '')}s {result.get('error', '')}", flush=True)


def cmd_finalize(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("text_cache", "depth_stats.json", "eef20_stats.json", "uvd_stats.json", "assets"):
        link = out / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(ORIGINAL / name)
    original_rows = [json.loads(l) for l in (ORIGINAL / "index.jsonl").read_text().splitlines() if l.strip()]
    if args.tasks:
        tasks = set(args.tasks.split(","))
        original_rows = [r for r in original_rows if r["split"] != "train" or r["task"] in tasks]
    for row in original_rows:
        task_dir = out / row["task"]
        task_dir.mkdir(exist_ok=True)
        link = task_dir / row["variant"]
        if not link.exists() and not link.is_symlink():
            link.symlink_to(ORIGINAL / row["task"] / row["variant"])
    instructions = (ORIGINAL / "episode_instructions_official.jsonl").read_text().splitlines()
    rollout_rows, successes = [], 0
    for meta_path in sorted(out.glob(f"*/rollout_{args.tag}/episode*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "ok":
            continue
        variant = meta_path.parent.parent.name
        row = {"task": meta["task"], "variant": variant, "episode": meta["episode"], "frames": meta["frames"],
               "split": "train", "success": meta["success"], "final_score": meta["final_score"], "rollout": True}
        rollout_rows += [row] * int(args.repeat)
        successes += int(meta["success"])
        instructions.append(json.dumps({"source": str(out / meta["task"] / variant / "data" / f"episode{meta['episode']}.hdf5"),
                                        "instructions": [meta["instruction"]]}))
    if args.include_dataset:
        previous = Path(args.include_dataset)
        old_rows = [json.loads(l) for l in (previous / "index.jsonl").read_text().splitlines() if l.strip()]
        seen = {(r["task"], r["variant"], r["episode"]) for r in rollout_rows}
        for row in old_rows:
            key = (row["task"], row["variant"], row["episode"])
            if not row.get("rollout") or key in seen:
                continue
            seen.add(key)
            task_dir = out / row["task"]
            task_dir.mkdir(exist_ok=True)
            link = task_dir / row["variant"]
            if not link.exists() and not link.is_symlink():
                link.symlink_to(previous / row["task"] / row["variant"])
            rollout_rows += [row] * int(args.repeat)
            successes += int(row["success"])
        instructions += (previous / "episode_instructions_official.jsonl").read_text().splitlines()
    with open(out / "index.jsonl", "w") as f:
        for row in original_rows + rollout_rows:
            f.write(json.dumps(row) + "\n")
    (out / "episode_instructions_official.jsonl").write_text("\n".join(instructions) + "\n")
    print(json.dumps({"original_rows": len(original_rows), "rollout_rows": len(rollout_rows), "rollout_episodes": len(rollout_rows) // max(args.repeat, 1),
                      "rollout_successes": successes, "repeat": args.repeat, "index": str(out / "index.jsonl")}))


def cmd_status(args) -> None:
    record = Path(args.record)
    counts = {"recordings": 0, "success": 0, "score_ge_50": 0, "frames": 0}
    per_task: dict[str, list[int]] = {}
    for path in list_recordings(record):
        with h5py.File(path) as rec:
            a = dict(rec.attrs)
        counts["recordings"] += 1
        counts["success"] += int(bool(a["success"]))
        counts["score_ge_50"] += int(float(a.get("final_score", 0)) >= 50)
        counts["frames"] += int(a["frames"])
        t = per_task.setdefault(str(a["task"]), [0, 0])
        t[0] += int(bool(a["success"]))
        t[1] += 1
    print(json.dumps(counts))
    for task, (s, n) in sorted(per_task.items()):
        print(f"{task:40s} {s}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--record", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tag", default="v1")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--tasks", help="comma-separated base task names")
    b.add_argument("--episode-offset", type=int, default=0, help="disjoint numbering for recordings from different seeds")
    b.add_argument("--min-score", type=float, default=50.0, help="process score (0-100) of a failed episode to keep it")
    b.add_argument("--successes-only", action="store_true")
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True)
    f.add_argument("--tag", default="v1")
    f.add_argument("--repeat", type=int, default=1, help="index rows per rollout episode (sampling weight)")
    f.add_argument("--tasks", help="keep official train rows for these tasks; retain all validation rows")
    f.add_argument("--include-dataset", help="retain unique rollouts from a previous round, with the new repeat weight")
    s = sub.add_parser("status")
    s.add_argument("--record", required=True)
    args = ap.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
