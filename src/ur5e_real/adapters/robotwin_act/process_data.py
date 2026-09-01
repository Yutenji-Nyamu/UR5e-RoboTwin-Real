from __future__ import annotations

import argparse
import json
from pathlib import Path

ACT_CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_right_wrist": "right_camera",
    "cam_left_wrist": "left_camera",
}


def load_episode(path: Path):
    import h5py

    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as root:
        action = root["joint_action"]
        arrays = {
            "left_gripper": action["left_gripper"][()],
            "left_arm": action["left_arm"][()],
            "right_gripper": action["right_gripper"][()],
            "right_arm": action["right_arm"][()],
        }
        images = {
            act_name: root[f"observation/{source_name}/rgb"][()]
            for act_name, source_name in ACT_CAMERA_MAP.items()
        }
    return arrays, images


def _decode_resize(encoded, size: tuple[int, int]):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode an episode image")
    return cv2.resize(image, size)


def transform_episode(source: Path, destination: Path, image_size: tuple[int, int] = (640, 480)) -> int:
    import h5py
    import numpy as np

    arrays, images = load_episode(source)
    left_gripper = np.asarray(arrays["left_gripper"]).reshape(-1, 1)
    right_gripper = np.asarray(arrays["right_gripper"]).reshape(-1, 1)
    left_arm = np.asarray(arrays["left_arm"])
    right_arm = np.asarray(arrays["right_arm"])
    length = min(len(left_gripper), len(right_gripper), len(left_arm), len(right_arm))
    if length < 2:
        raise ValueError(f"{source}: at least two timesteps are required")

    states = np.concatenate(
        [left_arm[:length], left_gripper[:length], right_arm[:length], right_gripper[:length]], axis=1
    ).astype(np.float32)
    qpos = states[:-1]
    actions = states[1:]
    output_length = len(qpos)
    decoded = {
        name: np.stack([_decode_resize(frame, image_size) for frame in values[:output_length]])
        for name, values in images.items()
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as output:
        output.create_dataset("action", data=actions)
        observations = output.create_group("observations")
        observations.create_dataset("qpos", data=qpos)
        observations.create_dataset("left_arm_dim", data=np.full(output_length, left_arm.shape[1]))
        observations.create_dataset("right_arm_dim", data=np.full(output_length, right_arm.shape[1]))
        image_group = observations.create_group("images")
        for name, values in decoded.items():
            image_group.create_dataset(name, data=values, dtype=np.uint8)
    return output_length


def update_task_config(act_root: Path, task_key: str, dataset_dir: Path, episodes: int) -> Path:
    config_path = act_root / "SIM_TASK_CONFIGS.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    try:
        relative_dataset = dataset_dir.relative_to(act_root)
        dataset_value = f"./{relative_dataset.as_posix()}"
    except ValueError:
        dataset_value = str(dataset_dir)
    config[task_key] = {
        "dataset_dir": dataset_value,
        "num_episodes": episodes,
        "episode_len": 1000,
        "camera_names": list(ACT_CAMERA_MAP),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def process_run(
    run_root: Path,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    episode_count: int,
) -> Path:
    act_root = robotwin_root / "policy" / "ACT"
    if not act_root.is_dir():
        raise FileNotFoundError(f"RoboTwin ACT checkout not found: {act_root}")
    raw_dir = run_root / task_name / task_config / "data"
    output_dir = act_root / "processed_data" / f"sim-{task_name}" / f"{task_config}-{episode_count}"
    for index in range(episode_count):
        source = raw_dir / f"episode{index}.hdf5"
        destination = output_dir / f"episode_{index}.hdf5"
        steps = transform_episode(source, destination)
        print(f"[SAVED] {destination} ({steps} transitions)")
    task_key = f"sim-{task_name}-{task_config}-{episode_count}"
    config_path = update_task_config(act_root, task_key, output_dir, episode_count)
    print(f"[CONFIG] {task_key} -> {config_path}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert real episodes into RoboTwin ACT training data")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--robotwin-root", type=Path, default=Path(".third_party/RoboTwin"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="simple")
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args(argv)
    process_run(args.run_root.resolve(), args.robotwin_root.resolve(), args.task, args.task_config, args.episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
