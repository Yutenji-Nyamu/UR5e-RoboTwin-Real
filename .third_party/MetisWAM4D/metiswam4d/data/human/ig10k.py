"""IG-10K human demonstration windows (fixed ego ZED2i, 512x288, 30 fps).

Per task directory (``scripts/data_prep/ig10k_anno_reader.py`` layout): LeRobot v3 videos, DA3 depth
(FFV1 12-bit), ``masks.h5`` (shipped instance ids, or SAM packbits channels for the ``_levels`` subset),
``track4d.h5`` (RAFT + DA3 object displacement, valid bits, full-frame flow) and, for the 53 MANO
directories, ``hand_track4d.h5``.  The camera is static, so the displacement field already follows the
Track4D definition and the camera tokens are supervised with the identity code.

Roles: hand / arm = 1 (body), manipulated objects = 2, everything else = 0 (black in the encoding,
as in the RoboTwin data).  Prompts: the English task paraphrases of ``_meta_extra/task_desc`` for the
``v1`` tasks, the level-specific task text for ``_levels``.

``profile="robot"`` reads the Realman dual-arm subset (``preprocessed/robot``, same layout, SAM arm
mask as role 1) and adds the action chunk / proprio in Alpha's unified 80-D space
(:mod:`metiswam4d.data.human.ig10k_action`).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from metiswam4d.data.camera import identity_code
from metiswam4d.data.human.ig10k_action import ACTION_COLUMN, STATE_COLUMN, IG10KActionNormalizer, encode_window_actions
from metiswam4d.data.human.window import HEIGHT, SOURCE_FRAMES, TRANSITIONS, WIDTH, video_frame_indices
from metiswam4d.data.rt2.codec import SHRINK_D_M, anchor_frame, encode_uvd, uvd_from_xyz

IG10K_ROOT = "/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/human"
IG10K_ROBOT_ROOT = "/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/robot"
TASK_DESC_DIR = "/ytech_milm_intern/danglingwei/datas/IG-10K-Dataset/_meta_extra/task_desc"
TASK_DESC = f"{TASK_DESC_DIR}/human_desc.json"
PROMPT_PREFIX = "A video recorded from a fixed camera of a person's hands executing the following task: "
ROBOT_PROMPT_PREFIX = "A video recorded from a fixed camera of a dual-arm robot executing the following task: "
ACTION_STEPS = 32


@dataclass
class IG10KConfig:
    profile: str = "human"                   # human | robot (robot: Realman dual-arm episodes with actions)
    root: str | None = None                  # defaults per profile
    manifest: str | None = None
    task_desc: str | None = None
    prompt_prefix: str | None = None
    action_stats: str | None = None          # robot: EEF20 min/max json (default <root>/action_stats_eef20.json)
    split: str = "train"                     # train | val | all
    val_episodes_per_task: int = 2           # the last N episodes of every task dir form the validation split
    samples_per_episode: int = 4
    limit: int | None = None
    seed: int = 0
    embodiment: int | None = None            # registry index (human_ego 0 / ig10k_realman 3)

    def __post_init__(self) -> None:
        if self.profile not in ("human", "robot"):
            raise ValueError("profile must be 'human' or 'robot'")
        robot = self.profile == "robot"
        self.root = self.root or (IG10K_ROBOT_ROOT if robot else IG10K_ROOT)
        self.manifest = self.manifest or f"{self.root}/manifest.parquet"
        self.task_desc = self.task_desc or f"{TASK_DESC_DIR}/{'robot' if robot else 'human'}_desc.json"
        self.prompt_prefix = self.prompt_prefix or (ROBOT_PROMPT_PREFIX if robot else PROMPT_PREFIX)
        if robot and self.action_stats is None:
            self.action_stats = f"{self.root}/action_stats_eef20.json"
        if self.embodiment is None:
            self.embodiment = 3 if robot else 0

    @property
    def with_action(self) -> bool:
        return self.profile == "robot"


def _episode_video_index(task_dir: Path, camera: str) -> dict[int, tuple[Path, float]]:
    """episode -> (video file, from_timestamp) from the LeRobot v3 episode tables."""
    index = {}
    for f in sorted((task_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["episode_index", f"videos/{camera}/chunk_index", f"videos/{camera}/file_index",
                                      f"videos/{camera}/from_timestamp"])
        for r in t.to_pylist():
            path = task_dir / "videos" / camera / f"chunk-{r[f'videos/{camera}/chunk_index']:03d}" \
                / f"file-{r[f'videos/{camera}/file_index']:03d}.mp4"
            index[int(r["episode_index"])] = (path, float(r[f"videos/{camera}/from_timestamp"]))
    return index


def _episode_data_index(task_dir: Path) -> dict[int, tuple[Path, int, int]]:
    """episode -> (data parquet, dataset_from_index, dataset_to_index) from the LeRobot v3 episode tables."""
    index = {}
    for f in sorted((task_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "data/chunk_index", "data/file_index",
                                      "dataset_from_index", "dataset_to_index"])
        for r in t.to_pylist():
            path = task_dir / "data" / f"chunk-{r['data/chunk_index']:03d}" / f"file-{r['data/file_index']:03d}.parquet"
            index[int(r["episode_index"])] = (path, int(r["dataset_from_index"]), int(r["dataset_to_index"]))
    return index


def read_episode_states(path: Path, episode: int, frames: list[int]) -> np.ndarray:
    """``[len(frames), 14]``: observation row at ``frames[0]`` followed by action rows ``frames[1:] - 1``
    (``action[t] == observation[t + 1]``), i.e. the Realman state at every listed frame of the episode."""
    t = pq.read_table(path, columns=["episode_index", "frame_index", STATE_COLUMN, ACTION_COLUMN])
    t = t.filter(pc.equal(t["episode_index"], episode))
    frame_index = t["frame_index"].to_numpy()
    order = np.argsort(frame_index)
    frame_index = frame_index[order]
    states = np.stack(t[STATE_COLUMN].to_numpy(zero_copy_only=False))[order].astype(np.float32)
    actions = np.stack(t[ACTION_COLUMN].to_numpy(zero_copy_only=False))[order].astype(np.float32)
    if frame_index[0] != 0 or np.any(np.diff(frame_index) != 1):
        raise ValueError(f"{path}: episode {episode} frame indices are not contiguous")
    if frames[-1] - 1 >= len(frame_index):
        raise ValueError(f"{path}: episode {episode} has {len(frame_index)} rows, window needs {frames[-1]}")
    rows = [states[frames[0]]] + [actions[k - 1] for k in frames[1:]]
    return np.stack(rows)


def decode_video_frames(path: Path, from_s: float, indices: list[int], fps: int = 30) -> list[np.ndarray]:
    """Frames ``indices`` (episode-relative) of an episode that starts at ``from_s`` inside ``path``."""
    import av
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Single-threaded and closed here: a threaded decoder left to the GC can block a DataLoader worker.
        stream.codec_context.thread_count = 1
        target = from_s + min(indices) / fps
        container.seek(int(max(target - 0.5, 0) / stream.time_base), stream=stream, backward=True, any_frame=False)
        decoded = container.decode(stream)
        try:
            for frame in decoded:
                if frame.time is None:
                    continue
                k = int(round((frame.time - from_s) * fps))
                if k in wanted and k not in out:
                    out[k] = frame.to_ndarray(format="rgb24")
                if k >= max(indices):
                    break
        finally:
            decoded.close()
    if len(out) != len(wanted):
        raise ValueError(f"{path}: decoded {len(out)} of {len(wanted)} frames from {from_s:.3f}s")
    return [out[k] for k in indices]


def decode_depth_frames(mkv: Path, indices: list[int], scale: float, offset: float) -> np.ndarray:
    """Metric depth (float32 [n, H, W]) at ``indices`` (ascending) of the FFV1 gray16 stream: one seek to the
    first frame, then sequential decode (intra-only stream, 30 fps, a window spans 33 frames)."""
    import av
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    with av.open(str(mkv)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = float(stream.average_rate or 30)
        container.seek(int(min(indices) / rate / stream.time_base), stream=stream, backward=True, any_frame=False)
        decoded = container.decode(stream)
        try:
            for frame in decoded:
                k = int(round(float(frame.pts * stream.time_base) * rate)) if frame.pts is not None else -1
                if k in wanted:
                    out[k] = frame.to_ndarray(format="gray16le").astype(np.float32) * scale + offset
                if k >= max(indices):
                    break
        finally:
            decoded.close()
    if len(out) != len(wanted):
        raise ValueError(f"{mkv}: decoded {len(out)} of {len(wanted)} depth frames")
    return np.stack([out[k] for k in indices])


def decode_depth_frame(mkv: Path, index: int, scale: float, offset: float) -> np.ndarray:
    return decode_depth_frames(mkv, [index], scale, offset)[0]


class IG10KWindowDataset(Dataset):
    def __init__(self, config: IG10KConfig):
        self.config = config
        rows = pq.read_table(config.manifest).to_pylist()
        rows = [r for r in rows if r["frames"] >= SOURCE_FRAMES and r["track4d_h5"]]
        by_task: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            by_task.setdefault((r["subset"], r["task"]), []).append(r)
        selected = []
        for key, group in sorted(by_task.items()):
            group.sort(key=lambda r: r["episode"])
            n_val = min(config.val_episodes_per_task, max(len(group) - 1, 0))
            val = group[len(group) - n_val:] if n_val else []
            train = group[:len(group) - n_val]
            selected += {"train": train, "val": val, "all": group}[config.split]
        if config.limit:
            selected = selected[:config.limit]
        if not selected:
            raise ValueError(f"IG10K split {config.split!r} is empty")
        self.rows = selected
        self.desc = json.loads(Path(config.task_desc).read_text()) if Path(config.task_desc).exists() else {}
        self.actions = IG10KActionNormalizer(config.action_stats) if config.with_action else None
        self._video_index: dict[tuple[str, str], dict] = {}
        self._data_index: dict[tuple[str, str], dict] = {}
        self._depth_index: dict[tuple[str, str], dict] = {}
        self._task_text: dict[tuple[str, str], list[str]] = {}
        self._log_dir = Path(config.root) / "_train_logs"

    def __len__(self) -> int:
        return len(self.rows) * self.config.samples_per_episode

    # ------------------------------------------------------------------ per-task caches
    def task_dir(self, row: dict) -> Path:
        return Path(self.config.root) / row["subset"] / row["task"]

    def video_of(self, row: dict) -> tuple[Path, float]:
        key = (row["subset"], row["task"])
        if key not in self._video_index:
            camera = Path(row["video"]).name.removeprefix("observation.images.")
            self._video_index[key] = _episode_video_index(self.task_dir(row), f"observation.images.{camera}")
        return self._video_index[key][int(row["episode"])]

    def data_of(self, row: dict) -> tuple[Path, int, int]:
        key = (row["subset"], row["task"])
        if key not in self._data_index:
            self._data_index[key] = _episode_data_index(self.task_dir(row))
        return self._data_index[key][int(row["episode"])]

    def depth_meta(self, row: dict) -> dict:
        key = (row["subset"], row["task"])
        if key not in self._depth_index:
            self._depth_index[key] = json.loads((self.task_dir(row) / "depth_index.json").read_text())["episodes"]
        return self._depth_index[key][f"ep_{int(row['episode']):05d}"]

    def prompts_of(self, row: dict) -> list[str]:
        key = (row["subset"], row["task"])
        if key not in self._task_text:
            # human_desc.json is keyed by the v1 task dir (H*_L*), robot_desc.json by the robot dir (robot_H*_L*);
            # the _levels human dirs have no entry and fall back to their tasks.parquet text + the base task.
            texts: list[str] = []
            if row["subset"] in ("imitator_human_v1", "imitator_robot_v1"):
                texts = list(self.desc.get(row["task"], []))
            if not texts:
                tasks = pq.read_table(self.task_dir(row) / "meta" / "tasks.parquet").to_pylist()
                texts = [str(t.get("task") or t.get("__index_level_0__") or row["task"]) for t in tasks]
                base = self.desc.get(row["task"].split("_L")[0], [])
                texts = [f"{t}. {b.split(';')[0].strip()}" for t in texts for b in base[:4]] or texts
            self._task_text[key] = [self.config.prompt_prefix + t for t in texts]
        return self._task_text[key]

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        for attempt in range(8):
            try:
                return self._load(index)
            except Exception as exc:  # noqa: BLE001
                self._record_bad(index, exc)
                index = (index + 7919 * (attempt + 1)) % len(self)
        raise RuntimeError("IG10KWindowDataset: 8 consecutive unreadable episodes")

    def _record_bad(self, index: int, exc: Exception) -> None:
        row = self.rows[index // self.config.samples_per_episode]
        try:
            self._log_dir.mkdir(exist_ok=True)
            with open(self._log_dir / "bad_episodes.jsonl", "a") as f:
                f.write(json.dumps({"subset": row["subset"], "task": row["task"], "episode": row["episode"],
                                    "error": repr(exc)[:200], "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
        print(f"[ig10k dataset] unreadable {row['subset']}/{row['task']}/ep{row['episode']}: {exc!r}", flush=True)

    def _masks(self, h: h5py.File, name: str, frames: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
        """(objects bool [n, H, W], arm bool [n, H, W] | None) at the given frames."""
        attrs = dict(h.attrs)
        d = h[name]
        idx = np.asarray(frames)
        if attrs.get("channels") == "instance_ids":
            ids = np.stack([d[t] for t in idx])
            if "arm_id" in attrs:  # robot subset: SAM-detected arm stored inside the id map
                arm = ids == int(attrs["arm_id"])
                return (ids > 0) & ~arm, arm
            return ids > 0, None
        chans = attrs["channels"].split(",")
        width = int(d.attrs["unpacked_shape"][-1])
        bits = np.unpackbits(np.stack([d[t] for t in idx]), axis=-1, count=width, bitorder="little").astype(bool)
        objects = bits[:, chans.index("manipulated_objects")]
        arm = bits[:, chans.index("arm_and_hand")] if "arm_and_hand" in chans else None
        return objects, arm

    def _load(self, index: int) -> dict:
        row = self.rows[index // self.config.samples_per_episode]
        rng = random.Random(hash((self.config.seed, index, torch.initial_seed())) & 0xFFFFFFFF)
        n = int(row["frames"])
        s = rng.randint(0, n - SOURCE_FRAMES)
        frames_idx = video_frame_indices(s)
        name = f"ep_{int(row['episode']):05d}"
        td = self.task_dir(row)

        path, from_s = self.video_of(row)
        frames = decode_video_frames(path, from_s, frames_idx)
        if frames[0].shape[:2] != (HEIGHT, WIDTH):
            raise ValueError(f"unexpected frame size {frames[0].shape}")
        meta = self.depth_meta(row)
        src_frames = frames_idx[:-1]  # transitions start at s + 4k
        depth = decode_depth_frames(td / "depth" / f"{name}.mkv", src_frames, float(meta["scale"]), float(meta["offset"]))

        with h5py.File(td / "track4d.h5") as f:
            K = np.asarray(f.attrs["K"], dtype=np.float64)
            g = f[name]
            src = g["source_frame_index"][:]
            rows_idx = [int(np.nonzero(src == t)[0][0]) for t in src_frames]
            delta = np.stack([g["delta_xyz_cam"][r] for r in rows_idx]).astype(np.float32) / 10000.0  # [8, H, W, 3] m
            valid = np.unpackbits(np.stack([g["valid_bits"][r] for r in rows_idx]), axis=-1, count=WIDTH,
                                  bitorder="little").astype(bool)
        hand = np.zeros_like(valid)
        hp = td / "hand_track4d.h5"
        if hp.exists():
            with h5py.File(hp) as f:
                if name in f:
                    g = f[name]
                    hsrc = g["source_frame_index"][:]
                    hrows = [int(np.nonzero(hsrc == t)[0][0]) for t in src_frames]
                    hdelta = np.stack([g["delta_xyz_cam"][r] for r in hrows]).astype(np.float32) / 10000.0
                    hand = np.unpackbits(np.stack([g["valid_bits"][r] for r in hrows]), axis=-1, count=WIDTH,
                                         bitorder="little").astype(bool)
                    delta[hand] = hdelta[hand]
                    valid |= hand
        with h5py.File(td / "masks.h5") as h:
            objects, arm = self._masks(h, name, src_frames)

        roles = np.zeros((TRANSITIONS, HEIGHT, WIDTH), dtype=np.uint8)
        if arm is not None:
            roles[arm & valid] = 1
        roles[objects & valid] = 2
        roles[hand] = 1  # MANO hand overrides the object mask where they overlap
        foreground = valid & (roles > 0)
        anchor_fg = foreground[0]
        uvd = uvd_from_xyz(delta, depth, K)                                             # [8, H, W, 3] (px, px, m)
        track_px, track_delta = encode_uvd(uvd, foreground, WIDTH, SHRINK_D_M["ig10k"])
        track_rgb = np.concatenate((anchor_frame(anchor_fg)[None], track_px))
        progress = torch.tensor([s / max(n - 1, 1)], dtype=torch.float32)
        sample: dict = {}
        if self.actions is not None:
            data_path, _, _ = self.data_of(row)
            states = read_episode_states(data_path, int(row["episode"]), list(range(s, s + ACTION_STEPS + 1)))
            action, action_mask, proprio, proprio_mask = encode_window_actions(states, self.actions)
            sample.update(action=torch.from_numpy(action), action_mask=torch.from_numpy(action_mask),
                          proprio=torch.from_numpy(proprio), proprio_mask=torch.from_numpy(proprio_mask))
        return {
            **sample,
            "key": f"ig10k/{row['subset']}/{row['task']}/{name}/s{s}",
            "prompt": rng.choice(self.prompts_of(row)),
            "embodiment": self.config.embodiment,
            "progress_video": progress, "progress_body": progress.clone(), "progress_valid": torch.tensor(True),
            "video_frames": torch.from_numpy(np.stack(frames)),
            "track_rgb": torch.from_numpy(track_rgb),
            "track_foreground": torch.from_numpy(np.concatenate((anchor_fg[None], foreground))),
            "track_role_px": torch.from_numpy(np.concatenate((roles[:1], roles))),
            "track_delta": torch.from_numpy(track_delta).to(torch.float16),
            "head_rgb": torch.from_numpy(frames[0]),
            "head_depth_m": torch.from_numpy(depth[0].astype(np.float32)),
            "depth_present": torch.tensor(True),
            "head_mask": torch.from_numpy(anchor_fg),
            "camera_delta": identity_code(TRANSITIONS),
            "camera_valid": torch.ones(TRANSITIONS, dtype=torch.bool),
        }


__all__ = ["ACTION_STEPS", "IG10KConfig", "IG10KWindowDataset", "decode_depth_frame", "decode_depth_frames",
           "decode_video_frames", "read_episode_states"]
