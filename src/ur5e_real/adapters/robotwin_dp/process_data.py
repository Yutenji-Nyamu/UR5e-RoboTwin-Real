from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_IMAGE_SIZE = (320, 240)  # width, height; RoboTwin D435 baseline


def motion_bounds(
    vectors,
    *,
    linear_threshold_m: float = 0.002,
    angular_threshold_rad: float = 0.0174533,
    lead_in_steps: int = 3,
    tail_steps: int = 3,
) -> tuple[int, int]:
    """Keep the active interval plus short stationary context at both ends."""
    import numpy as np

    values = np.asarray(vectors)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError("state vectors must have shape (steps, 14)")
    if min(linear_threshold_m, angular_threshold_rad, lead_in_steps, tail_steps) < 0:
        raise ValueError("motion thresholds, lead-in, and tail must be non-negative")
    if len(values) < 2:
        return 0, len(values)

    def changed_from(reference):
        translated = np.linalg.norm(values[:, :3] - reference[:3], axis=1) > linear_threshold_m
        rotated = np.linalg.norm(values[:, 3:6] - reference[3:6], axis=1) > angular_threshold_rad
        gripper_changed = np.abs(values[:, 13] - reference[13]) > 0.5
        return translated | rotated | gripper_changed

    origin = values[0]
    start_candidates = np.flatnonzero(changed_from(origin))
    if not len(start_candidates):
        return 0, len(values)
    start = max(0, int(start_candidates[0]) - lead_in_steps)

    end_candidates = np.flatnonzero(changed_from(values[-1]))
    stop = (
        min(len(values), int(end_candidates[-1]) + 1 + tail_steps)
        if len(end_candidates)
        else len(values)
    )
    return (start, stop) if stop - start >= 2 else (0, len(values))


def _decode_resize(encoded, image_size: tuple[int, int]):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode an episode image")
    return cv2.resize(image, image_size)


def _load_episode(path: Path):
    import h5py
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as root:
        action = root["joint_action"]
        if "vector" in action:
            vectors = action["vector"][()].astype(np.float32)
        else:
            vectors = np.concatenate(
                (
                    action["left_arm"][()],
                    np.asarray(action["left_gripper"][()]).reshape(-1, 1),
                    action["right_arm"][()],
                    np.asarray(action["right_gripper"][()]).reshape(-1, 1),
                ),
                axis=1,
            ).astype(np.float32)
        head_images = root["observation/head_camera/rgb"][()]
        source_run_id = str(root.attrs.get("source_run_id", path.stem))
        schema_version = int(root.attrs.get("schema_version", 1))
    return vectors, head_images, source_run_id, schema_version


def _append(dataset, values) -> None:
    start = dataset.shape[0]
    end = start + len(values)
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values


def process_run(
    run_root: Path,
    task_name: str,
    task_config: str,
    episode_count: int,
    *,
    output_path: Path | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    overwrite: bool = False,
    trim_static_edges: bool = False,
    motion_threshold_mm: float = 2.0,
    motion_threshold_deg: float = 1.0,
    lead_in_steps: int = 3,
    tail_steps: int = 3,
) -> Path:
    import numpy as np
    import zarr

    run_root = run_root.expanduser().resolve()
    source_dir = run_root / task_name / task_config / "data"
    if output_path is None:
        output_path = run_root.parent / "dp" / f"{task_name}-{task_config}-{episode_count}.zarr"
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = image_size
    root = zarr.open_group(str(output_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    head_store = data.create_dataset(
        "head_camera",
        shape=(0, 3, height, width),
        chunks=(32, 3, height, width),
        dtype="uint8",
        compressor=compressor,
    )
    state_store = data.create_dataset(
        "state", shape=(0, 14), chunks=(100, 14), dtype="float32", compressor=compressor
    )
    action_store = data.create_dataset(
        "action", shape=(0, 14), chunks=(100, 14), dtype="float32", compressor=compressor
    )

    episode_ends: list[int] = []
    source_run_ids: list[str] = []
    source_schema_versions: list[int] = []
    trim_start_indices: list[int] = []
    trim_stop_indices: list[int] = []
    source_lengths: list[int] = []
    total = 0
    for episode_index in range(episode_count):
        source = source_dir / f"episode{episode_index}.hdf5"
        vectors, images, run_id, schema_version = _load_episode(source)
        length = min(len(vectors), len(images))
        if length < 2:
            raise ValueError(f"{source}: at least two timesteps are required")
        trim_start, trim_stop = (
            motion_bounds(
                vectors[:length],
                linear_threshold_m=motion_threshold_mm / 1000.0,
                angular_threshold_rad=np.deg2rad(motion_threshold_deg),
                lead_in_steps=lead_in_steps,
                tail_steps=tail_steps,
            )
            if trim_static_edges
            else (0, length)
        )
        if trim_stop - trim_start < 2:
            raise ValueError(f"{source}: fewer than two timesteps remain after edge trimming")
        head = np.stack(
            [_decode_resize(frame, image_size) for frame in images[trim_start : trim_stop - 1]]
        )
        head = np.moveaxis(head, -1, 1).astype(np.uint8)
        _append(head_store, head)
        _append(state_store, vectors[trim_start : trim_stop - 1])
        _append(action_store, vectors[trim_start + 1 : trim_stop])
        transitions = trim_stop - trim_start - 1
        total += transitions
        episode_ends.append(total)
        source_run_ids.append(run_id)
        source_schema_versions.append(schema_version)
        trim_start_indices.append(trim_start)
        trim_stop_indices.append(trim_stop)
        source_lengths.append(length)
        print(
            f"[DP] episode {episode_index}: {transitions} transitions "
            f"(trimmed head={trim_start}, tail={length - trim_stop}, {run_id})"
        )

    meta.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        dtype="int64",
        compressor=compressor,
    )
    root.attrs.update(
        {
            "format": "robotwin_dp_real_v1",
            "task": task_name,
            "task_config": task_config,
            "selection": "outcome=success",
            "source_run_ids": source_run_ids,
            "source_schema_versions": source_schema_versions,
            "trim_static_edges": trim_static_edges,
            "trim_start_indices": trim_start_indices,
            "trim_stop_indices": trim_stop_indices,
            "source_lengths": source_lengths,
            "motion_threshold_mm": motion_threshold_mm,
            "motion_threshold_deg": motion_threshold_deg,
            "lead_in_steps": lead_in_steps,
            "tail_steps": tail_steps,
            "state_layout": "tcp6,dummy_gripper,tcp6,physical_gripper",
            "action_semantics": "next_absolute_state",
            "image_size_wh": [width, height],
            "source": json.dumps({"run_root": str(run_root)}, ensure_ascii=False),
        }
    )
    print(f"[SAVED] {output_path} ({total} transitions, {episode_count} episodes)")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert real HDF5 episodes into RoboTwin DP Zarr")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="simple")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trim-static-edges", action="store_true")
    parser.add_argument("--motion-threshold-mm", type=float, default=2.0)
    parser.add_argument("--motion-threshold-deg", type=float, default=1.0)
    parser.add_argument("--lead-in-steps", type=int, default=3)
    parser.add_argument("--tail-steps", type=int, default=3)
    args = parser.parse_args(argv)
    process_run(
        args.run_root,
        args.task,
        args.task_config,
        args.episodes,
        output_path=args.output,
        overwrite=args.overwrite,
        trim_static_edges=args.trim_static_edges,
        motion_threshold_mm=args.motion_threshold_mm,
        motion_threshold_deg=args.motion_threshold_deg,
        lead_in_steps=args.lead_in_steps,
        tail_steps=args.tail_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
