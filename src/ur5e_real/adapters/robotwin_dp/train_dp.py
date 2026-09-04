from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    task_name: str
    task_config: str
    episode_count: int
    seed: int = 0
    gpu: str = "0"
    debug: bool = False
    batch_size: int | None = None
    val_ratio: float | None = None


def default_output_dir(zarr_path: Path, train: TrainConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if zarr_path.parent.name == "dp" and zarr_path.parent.parent.name == "converted":
        root = zarr_path.parent.parent.parent
        return root / "checkpoints" / "dp" / f"{zarr_path.stem}-seed{train.seed}-{stamp}"
    return zarr_path.parent / "outputs" / f"{zarr_path.stem}-seed{train.seed}-{stamp}"


def build_train_command(
    robotwin_root: Path,
    zarr_path: Path,
    output_dir: Path,
    train: TrainConfig,
) -> list[str]:
    import zarr

    dp_root = robotwin_root / "policy" / "DP"
    train_script = dp_root / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"RoboTwin DP train.py not found: {train_script}")
    dataset = zarr.open_group(str(zarr_path), mode="r")
    transitions = int(dataset["data/state"].shape[0])
    if transitions < 1:
        raise ValueError(f"empty DP dataset: {zarr_path}")
    if "meta/episode_ends" in dataset:
        episode_ends = [int(value) for value in dataset["meta/episode_ends"][:]]
        episode_starts = [0, *episode_ends[:-1]]
        shortest_episode = min(end - start for start, end in zip(episode_starts, episode_ends))
    else:
        shortest_episode = transitions
    # RoboTwin drops incomplete batches for both train and validation. Keeping
    # the automatic batch within one episode guarantees a validation batch for
    # small real datasets while retaining the upstream batch 128 when possible.
    batch_size = train.batch_size or min(128, shortest_episode)
    val_ratio = train.val_ratio if train.val_ratio is not None else (0.0 if train.episode_count == 1 else 0.02)
    experiment = f"{train.task_name}-robot_dp-real"
    return [
        os.fspath(Path(os.sys.executable)),
        os.fspath(train_script),
        "--config-name=robot_dp_14.yaml",
        f"task.name={train.task_name}",
        f"task.dataset.zarr_path={zarr_path}",
        f"task.dataset.val_ratio={val_ratio}",
        f"task.dataset.batch_size={batch_size}",
        f"dataloader.batch_size={batch_size}",
        f"val_dataloader.batch_size={batch_size}",
        f"training.debug={str(train.debug).lower()}",
        f"training.seed={train.seed}",
        "training.device=cuda:0",
        "training.resume=false",
        f"exp_name={experiment}",
        "logging.mode=offline",
        f"setting={train.task_config}",
        f"expert_data_num={train.episode_count}",
        "head_camera_type=D435",
        f"hydra.run.dir={output_dir}",
    ]


def run_training(
    robotwin_root: Path,
    zarr_path: Path,
    train: TrainConfig,
    *,
    output_dir: Path | None = None,
) -> Path:
    robotwin_root = robotwin_root.expanduser().resolve()
    zarr_path = zarr_path.expanduser().resolve()
    if not zarr_path.is_dir():
        raise FileNotFoundError(zarr_path)
    output_dir = (output_dir or default_output_dir(zarr_path, train)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(robotwin_root, zarr_path, output_dir, train)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = train.gpu
    environment["HYDRA_FULL_ERROR"] = "1"
    print(f"[TRAIN] {shlex.join(command)}", flush=True)
    # RoboTwin names checkpoints with a relative path, so use the run directory
    # as cwd while keeping its model/training code untouched.
    subprocess.run(command, cwd=output_dir, env=environment, check=True)
    print(f"[OUTPUT] {output_dir}", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train RoboTwin DP on a real-robot Zarr dataset")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--robotwin-root", type=Path, default=Path(".third_party/RoboTwin"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="simple")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    run_training(
        args.robotwin_root,
        args.zarr_path,
        TrainConfig(
            task_name=args.task,
            task_config=args.task_config,
            episode_count=args.episodes,
            seed=args.seed,
            gpu=args.gpu,
            debug=args.debug,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
        ),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
