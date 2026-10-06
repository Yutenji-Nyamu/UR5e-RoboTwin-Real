#!/usr/bin/env python
"""Uniform reader for the IG-10K human annotation set produced by ig10k_human_depth_masks.py.

The on-disk layout is deliberately "use what shipped, estimate what did not", so the 69 task
dirs are not homogeneous:

  * 53 `imitator_human_v1` dirs      masks = IG-10K's own instance-id stream (uint8 id map),
                                      hands = MANO columns in the LeRobot data parquet
  * 16 `imitator_human_v1_levels`    masks = SAM 3.1 packbits channels (objects, + arm where
                                      no MANO exists)
  * all 69                            depth = DA3 metric depth, 12-bit FFV1 (lossless)
  * 210 `imitator_robot_v1` (profile robot, separate root)
                                      masks = shipped instance ids + SAM 3.1 robot arm stored as
                                      id `arm_id` in the same map; actions in data/*.parquet

This module hides that: every episode comes back as
    depth   float32 (T, 288, 512)  metres, camera frame
    objects bool    (T, 288, 512)  union of manipulated objects
    ids     uint8   (T, 288, 512)  per-instance ids when shipped, else 0/1 from `objects`
    arm     bool    (T, 288, 512) | None   only where SAM produced it
    labels  {id: name}
    mano    list[str]              MANO column names in the source parquet (may be empty)

Usage
    from ig10k_anno_reader import IG10KAnno
    anno = IG10KAnno()                 # human set;  IG10KAnno(profile="robot") for the robot set
    ep = anno.load("imitator_human_v1", "human_H1", 0)
    ep.depth.shape, ep.objects.mean(), ep.labels

    python ig10k_anno_reader.py manifest [human|robot]   # one parquet index over all episodes
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_profiles as PROF  # noqa: E402

ANNO_ROOT = PROF.PROFILES["human"]["out_root"]
VIDEO_ROOT = ANNO_ROOT  # one root: videos, MANO params and annotations live together per task dir
SUBSETS = PROF.PROFILES["human"]["subsets"]


@dataclass
class Episode:
    subset: str
    task: str
    episode: int
    frames: int
    depth: np.ndarray | None
    objects: np.ndarray
    ids: np.ndarray
    arm: np.ndarray | None
    labels: dict[int, str]
    mano: list[str] = field(default_factory=list)
    from_s: float = 0.0
    to_s: float = 0.0
    mask_source: str = ""
    depth_scale: float = 0.0
    depth_offset: float = 0.0


class IG10KAnno:
    def __init__(self, anno_root: Path | None = None, video_root: Path | None = None,
                 profile: str = "human"):
        self.profile = PROF.profile(profile)
        self.anno_root = Path(anno_root or self.profile["out_root"])
        self.video_root = Path(video_root or self.anno_root)
        self.subsets = self.profile["subsets"]

    # ------------------------------------------------------------------ paths
    def task_dir(self, subset: str, task: str) -> Path:
        return self.anno_root / subset / task

    def depth_index(self, subset: str, task: str) -> dict:
        return json.loads((self.task_dir(subset, task) / "depth_index.json").read_text())

    def episode_ids(self, subset: str, task: str) -> list[int]:
        import h5py
        with h5py.File(self.task_dir(subset, task) / "masks.h5") as h:
            return sorted(int(k.split("_")[1]) for k in h)

    # ------------------------------------------------------------------ depth
    def load_depth(self, subset: str, task: str, episode: int) -> np.ndarray:
        idx = self.depth_index(subset, task)
        name = f"ep_{episode:05d}"
        e = idx["episodes"][name]
        mkv = self.task_dir(subset, task) / "depth" / f"{name}.mkv"
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mkv), "-f", "rawvideo",
                            "-pix_fmt", "gray16le", "-"], capture_output=True, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"depth decode failed for {mkv}: {p.stderr.decode()[:200]}")
        q = np.frombuffer(p.stdout, dtype=np.uint16).reshape(-1, e["height"], e["width"])
        if q.shape[0] != e["frames"]:
            raise RuntimeError(f"{mkv}: decoded {q.shape[0]} frames, index says {e['frames']}")
        return q.astype(np.float32) * e["scale"] + e["offset"]

    # ------------------------------------------------------------------ masks
    def load_masks(self, subset: str, task: str, episode: int):
        """-> (objects bool, ids uint8, arm bool|None, labels, attrs)"""
        import h5py
        name = f"ep_{episode:05d}"
        with h5py.File(self.task_dir(subset, task) / "masks.h5") as h:
            a = dict(h.attrs)
            d = h[name]
            da = dict(d.attrs)
            if a.get("channels") == "instance_ids":
                ids = d[:].astype(np.uint8)
                labels = {int(k): v for k, v in json.loads(a["instance_labels"]).items()}
                arm = None
                objects = ids > 0
                if "arm_id" in a:  # robot subset: SAM arm lives in the id map
                    arm = ids == int(a["arm_id"])
                    objects &= ~arm
                return objects, ids, arm, labels, {**a, **da}
            # SAM packbits path
            n, c, hgt, wid = (int(x) for x in da["unpacked_shape"])
            bits = np.unpackbits(d[:], axis=-1, count=wid, bitorder="little").astype(bool)
            chans = a["channels"].split(",")
            arm = bits[:, chans.index("arm_and_hand")] if "arm_and_hand" in chans else None
            objects = bits[:, chans.index("manipulated_objects")]
            labels = {1: " | ".join(json.loads(da.get("prompts_kept", "[]")))}
            return objects, objects.astype(np.uint8), arm, labels, {**a, **da}

    # ------------------------------------------------------------------ track4d
    def load_track4d(self, subset: str, task: str, episode: int, with_hands: bool = True):
        """Dense camera-frame 3D displacement for every stride-4 pair of the episode.

        Returns dict(src=int32 (P,), delta=float32 (P,H,W,3) metres, valid=bool (P,H,W),
                     hand=bool (P,H,W) | None, K=(3,3)).
        Object pixels come from track4d.h5 (RAFT flow lifted through DA3 depth); where
        hand_track4d.h5 exists (the 53 MANO dirs) the MANO-mesh hand displacement is written
        over the hand's footprint and `hand` marks those pixels.  Both live in the same DA3
        camera frame with the same K, so the union is one consistent field.
        """
        import h5py
        name = f"ep_{episode:05d}"
        td = self.task_dir(subset, task)
        with h5py.File(td / "track4d.h5") as f:
            K = np.asarray(f.attrs["K"], np.float32)
            g = f[name]
            src = g["source_frame_index"][:]
            delta = g["delta_xyz_cam"][:].astype(np.float32) / 10000.0
            w = delta.shape[2]
            valid = np.unpackbits(g["valid_bits"][:], axis=-1, count=w, bitorder="little").astype(bool)
        hand = None
        hp = td / "hand_track4d.h5"
        if with_hands and hp.exists():
            with h5py.File(hp) as f:
                if name in f:
                    g = f[name]
                    hsrc = g["source_frame_index"][:]
                    hdelta = g["delta_xyz_cam"][:].astype(np.float32) / 10000.0
                    hvalid = np.unpackbits(g["valid_bits"][:], axis=-1, count=w, bitorder="little").astype(bool)
                    if not np.array_equal(hsrc, src):
                        raise RuntimeError(f"{task}/{name}: hand/object pair indices differ")
                    hand = hvalid
                    delta[hvalid] = hdelta[hvalid]
                    valid |= hvalid
        return {"src": src, "delta": delta, "valid": valid, "hand": hand, "K": K}

    # ------------------------------------------------------------------ one call
    def load(self, subset: str, task: str, episode: int, with_depth: bool = True) -> Episode:
        objects, ids, arm, labels, attrs = self.load_masks(subset, task, episode)
        depth = self.load_depth(subset, task, episode) if with_depth else None
        if depth is not None and depth.shape != objects.shape:
            raise RuntimeError(f"depth {depth.shape} vs mask {objects.shape} mismatch")
        e = self.depth_index(subset, task)["episodes"].get(f"ep_{episode:05d}", {})
        return Episode(
            subset=subset, task=task, episode=episode, frames=int(objects.shape[0]),
            depth=depth, objects=objects, ids=ids, arm=arm, labels=labels,
            mano=json.loads(attrs.get("mano_columns", "[]")),
            from_s=float(attrs.get("from_s", 0.0)), to_s=float(attrs.get("to_s", 0.0)),
            mask_source=str(attrs.get("object_source", "")),
            depth_scale=float(e.get("scale", 0.0)), depth_offset=float(e.get("offset", 0.0)),
        )

    # ------------------------------------------------------------------ manifest
    def manifest(self, out: Path | None = None):
        """One row per episode across all task dirs: paths + what is available."""
        import h5py
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for subset in self.subsets:
            for td in sorted((self.anno_root / subset).glob("*")):
                if not PROF.allowed(self.profile, td.name):
                    continue
                if not (td / "masks.h5").exists() or not (td / "depth_index.json").exists():
                    continue
                idx = self.depth_index(subset, td.name)
                data_files = sorted((self.video_root / subset / td.name / "data").glob("chunk-*/*.parquet"))
                action_cols = [c for c in pq.read_schema(data_files[0]).names
                               if c.startswith("action.")] if data_files else []
                with h5py.File(td / "masks.h5") as h:
                    a = dict(h.attrs)
                    for k in h:
                        ep = int(k.split("_")[1])
                        e = idx["episodes"].get(k)
                        if e is None:
                            continue
                        rows.append({
                            "subset": subset, "task": td.name, "episode": ep,
                            "frames": int(e["frames"]),
                            "from_s": float(e["from_s"]), "to_s": float(e["to_s"]),
                            "video": str(self.video_root / subset / td.name / "videos"
                                         / f"observation.images.{idx['ego_key']}"),
                            "depth_mkv": str(td / "depth" / f"{k}.mkv"),
                            "masks_h5": str(td / "masks.h5"),
                            "mask_format": a.get("channels", ""),
                            "has_arm_mask": "arm_and_hand" in a.get("channels", "") or "arm_id" in a,
                            "has_mano": bool(json.loads(a.get("mano_columns", "[]"))),
                            "depth_source": idx["model"],
                            "track4d_h5": str(td / "track4d.h5") if (td / "track4d.h5").exists() else "",
                            "hand_track4d_h5": str(td / "hand_track4d.h5") if (td / "hand_track4d.h5").exists() else "",
                            "data_dir": str(self.video_root / subset / td.name / "data"),
                            "action_columns": json.dumps(action_cols),
                        })
        t = pa.Table.from_pylist(rows)
        out = out or (self.anno_root / "manifest.parquet")
        pq.write_table(t, out)
        return t, out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manifest":
        t, out = IG10KAnno(profile=sys.argv[2] if len(sys.argv) > 2 else "human").manifest()
        import pyarrow.compute as pc
        print(f"{t.num_rows} episodes, {pc.sum(t['frames']).as_py()} frames -> {out}")
        for col in ("mask_format", "has_arm_mask", "has_mano"):
            print(f"  {col}: {dict(zip(*[x.to_pylist() for x in pc.value_counts(t[col]).flatten()]))}")
    else:
        a = IG10KAnno()
        ep = a.load("imitator_human_v1", "human_H1", 0)
        print(ep.frames, ep.depth.shape, ep.depth.min(), ep.depth.max(), ep.objects.mean(),
              ep.labels, len(ep.mano), ep.mask_source[:60])
