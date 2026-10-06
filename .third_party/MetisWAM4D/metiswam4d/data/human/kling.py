"""KlingHumanEgo-2.5M-5000H windows (30 fps transcodes in tar shards + sparse MANO hand fits + camera trajectories).

Two modes share one dataset class:

``track``  video + hand Track4D.  Rows are the hand-track subset at native 512x288 (the MANO cameras are
           calibrated in that pixel frame); the window start is drawn among starts whose 8 transitions
           contain at least ``min_pairs`` tracked hand pairs, so the Track block is never empty.
``video``  video only (any row that passes the aspect-ratio filter; frames are centre-cropped / resized).

Every item decodes the clip from its shard by byte offset (PyAV) and, in ``track`` mode, builds the
painted hand meshes of the window with :mod:`hand_render`; the online encoder rasterises them on the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from metiswam4d.data.human.hand_render import hand_window_meshes, transition_pairs, window_coverage
from metiswam4d.data.human.window import (
    FPS, HEIGHT, SOURCE_FRAMES, STRIDE, TRANSITIONS, VIDEO_FRAMES, WIDTH, fit_frame, source_frame_map,
    video_frame_indices,
)

KLING_ROOT = "/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M"


@dataclass
class KlingConfig:
    parquet: str = f"{KLING_ROOT}/KlingHumanEgo20M_30fps_hi_handtrack_subset.parquet"
    mode: str = "track"                      # track | video
    min_pairs: int = 4                       # transitions (of 8) with a tracked hand pair for a track window
    min_aspect: float = 1.5                  # video mode: keep 1.5 <= w/h <= 2.0, fit to 512x288
    max_aspect: float = 2.0
    camera_world_to_cam: bool = True         # extrinsics convention of the camera npz
    camera_compensate: bool = True           # express the target-frame hand in the source-frame camera axes
    camera_use_translation: bool = False     # trust the metric scale of the trajectory translation
    camera_min_conf: float = 0.0             # per-frame confidence threshold for the camera label
    samples_per_clip: int = 1                # dataset length multiplier (windows are drawn per item)
    limit: int | None = None                 # debug: first N rows
    seed: int = 0
    embodiment: int = 0                      # registry index of human_ego

    def __post_init__(self) -> None:
        if self.mode not in ("track", "video"):
            raise ValueError("mode must be 'track' or 'video'")


_COLUMNS = ["blobstore_key", "caption", "video_path", "offset", "size", "width", "height", "nb_frames", "src_fps",
            "src_nb_frames", "mano_path", "mano_offset", "mano_size", "camera_npz_archive", "camera_npz_member"]


def window_prompt(caption: str | None, t0: float, t1: float) -> str:
    """Kling v4 captions are JSON ``{"global": ..., "events": [{start_sec, end_sec, content}]}``: the prompt of
    a window is the global description plus the events overlapping ``[t0, t1]`` seconds."""
    if not caption:
        return ""
    try:
        doc = json.loads(caption)
    except (json.JSONDecodeError, TypeError):
        return str(caption)
    if not isinstance(doc, dict):
        return str(caption)
    parts = [str(doc.get("global", "")).strip()]
    for ev in doc.get("events", []) or []:
        try:
            if float(ev.get("end_sec", 0)) > t0 and float(ev.get("start_sec", 0)) < t1:
                parts.append(str(ev.get("content", "")).strip())
        except (TypeError, ValueError):
            continue
    return " ".join(p for p in parts if p)


def read_bytes(path: str, offset: int, size: int) -> bytes:
    with open(path, "rb") as fh:
        fh.seek(int(offset))
        return fh.read(int(size))


def decode_frames(data: bytes, indices: list[int]) -> list[np.ndarray]:
    """Sequential decode up to the last requested frame (clips are a few seconds)."""
    import av
    wanted = set(indices)
    last = max(indices)
    out: dict[int, np.ndarray] = {}
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        # Single-threaded and closed here: a threaded decoder left to the GC can block a DataLoader worker.
        stream.codec_context.thread_count = 1
        decoded = container.decode(stream)
        try:
            for k, frame in enumerate(decoded):
                if k in wanted:
                    out[k] = frame.to_ndarray(format="rgb24")
                if k >= last:
                    break
        finally:
            decoded.close()
    if len(out) != len(wanted):
        raise ValueError(f"decoded {len(out)} of {len(wanted)} requested frames (clip has fewer frames than indexed)")
    return [out[k] for k in indices]


class KlingWindowDataset(Dataset):
    def __init__(self, config: KlingConfig):
        self.config = config
        table = pq.read_table(config.parquet, columns=_COLUMNS)
        w, h = table["width"], table["height"]
        if config.mode == "track":
            keep = pc.and_(pc.equal(w, WIDTH), pc.equal(h, HEIGHT))
            keep = pc.and_(keep, pc.greater(table["mano_size"], 0))
        else:
            ratio = pc.divide(pc.cast(w, pa.float64()), pc.cast(h, pa.float64()))
            keep = pc.and_(pc.greater_equal(ratio, config.min_aspect), pc.less_equal(ratio, config.max_aspect))
        keep = pc.and_(keep, pc.greater_equal(table["nb_frames"], SOURCE_FRAMES))
        table = table.filter(keep)
        if config.limit:
            table = table.slice(0, config.limit)
        self.table = table.combine_chunks()
        self._log_dir = Path(KLING_ROOT) / "_train_logs"

    def __len__(self) -> int:
        return self.table.num_rows * self.config.samples_per_clip

    def row(self, index: int) -> dict:
        i = index % self.table.num_rows
        return {name: self.table[name][i].as_py() for name in _COLUMNS}

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        for attempt in range(8):
            try:
                return self._load(index)
            except Exception as exc:  # noqa: BLE001 - a bad clip must not kill the run
                self._record_bad(index, exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("KlingWindowDataset: 8 consecutive unreadable clips")

    def _record_bad(self, index: int, exc: Exception) -> None:
        try:
            self._log_dir.mkdir(exist_ok=True)
            row = self.row(index)
            with open(self._log_dir / f"bad_clips_{self.config.mode}.jsonl", "a") as f:
                f.write(json.dumps({"key": row["blobstore_key"], "error": repr(exc)[:200],
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[kling dataset] unreadable item {index}: {exc!r}", flush=True)

    def _load(self, index: int) -> dict:
        row = self.row(index)
        rng = random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)
        n = int(row["nb_frames"])
        key = row["blobstore_key"].split(":")[-1]
        # Clips are arbitrary cuts: no task phase.  Zero targets with progress_valid=False (masked in the loss).
        sample: dict = {"key": f"kling/{key}", "embodiment": self.config.embodiment,
                        "progress_video": torch.zeros(1), "progress_body": torch.zeros(1),
                        "progress_valid": torch.tensor(False)}

        def finish(s: int) -> None:
            sample["key"] += f"/s{s}"
            sample["prompt"] = window_prompt(row["caption"], s / FPS, (s + SOURCE_FRAMES) / FPS)

        if self.config.mode == "video":
            s = rng.randint(0, n - SOURCE_FRAMES)
            frames = decode_frames(read_bytes(row["video_path"], row["offset"], row["size"]), video_frame_indices(s))
            sample["video_frames"] = torch.from_numpy(np.stack([fit_frame(f) for f in frames]))
            finish(s)
            return sample

        fit = dict(np.load(io.BytesIO(read_bytes(row["mano_path"], row["mano_offset"], row["mano_size"]))))
        smap = source_frame_map(n, row["src_fps"], int(row["src_nb_frames"]))
        coverage = window_coverage(transition_pairs(fit, smap, n), n)
        good = np.nonzero(coverage >= self.config.min_pairs)[0]
        if len(good) == 0:
            good = np.nonzero(coverage >= max(int(coverage.max()), 1))[0]
        if len(good) == 0:
            raise ValueError("no window with a tracked hand pair")
        s = int(good[rng.randrange(len(good))])

        intrinsics = extrinsics = conf = None
        try:
            cam = load_camera_npz(row["camera_npz_archive"], row["camera_npz_member"])
            intrinsics = np.asarray(cam["intrinsics"][0], dtype=np.float64)
            extrinsics = np.asarray(cam["extrinsics"], dtype=np.float64)
            conf = np.asarray(cam.get("confidence_median_per_frame", np.ones(len(extrinsics))), dtype=np.float32)
        except (KeyError, OSError, tarfile.TarError, ValueError):
            pass  # no trajectory: uncompensated hand motion, camera tokens unsupervised
        hand = hand_window_meshes(
            fit, start=s, smap=smap, width=WIDTH, height=HEIGHT, intrinsics=intrinsics,
            extrinsics=extrinsics, world_to_cam=self.config.camera_world_to_cam,
            compensate=self.config.camera_compensate, use_translation=self.config.camera_use_translation,
            camera_conf=conf, camera_min_conf=self.config.camera_min_conf)
        frames = decode_frames(read_bytes(row["video_path"], row["offset"], row["size"]), video_frame_indices(s))
        finish(s)
        # Track pixels (track_rgb / foreground / roles / delta / head_mask) are rasterised from these meshes on
        # the GPU by the online encoder (hand_render.hand_pixels).
        sample.update({
            "video_frames": torch.from_numpy(np.stack(frames)),                    # uint8 [9, 288, 512, 3]
            "hand_verts": torch.from_numpy(hand.verts),                            # [9, 2, 778, 3] NDC
            "hand_colors": torch.from_numpy(hand.colors),                          # [9, 2, 778, 3]
            "hand_is_right": torch.from_numpy(hand.is_right),                      # bool [9, 2]
            "hand_present": torch.from_numpy(hand.present),                        # bool [9, 2]
            "head_rgb": torch.from_numpy(frames[0]),
            "head_depth_m": torch.zeros(HEIGHT, WIDTH, dtype=torch.float32),
            "depth_present": torch.tensor(False),
            "camera_delta": torch.from_numpy(hand.camera_delta),                   # [8, 9]
            "camera_valid": torch.from_numpy(hand.camera_valid),                   # [8]
            "track_pairs": torch.from_numpy(hand.pairs_per_transition),
            "track_saturation": torch.tensor(hand.saturation, dtype=torch.float32),
        })
        return sample


_TARS: dict[str, tarfile.TarFile] = {}


def load_camera_npz(archive: str, member: str) -> dict:
    tf = _TARS.get(archive)
    if tf is None:
        if len(_TARS) >= 16:
            for t in _TARS.values():
                t.close()
            _TARS.clear()
        tf = _TARS[archive] = tarfile.open(archive)
    handle = tf.extractfile(member)
    if handle is None:
        raise KeyError(member)
    return dict(np.load(io.BytesIO(handle.read())))


__all__ = ["KlingConfig", "KlingWindowDataset", "decode_frames", "load_camera_npz", "read_bytes"]
