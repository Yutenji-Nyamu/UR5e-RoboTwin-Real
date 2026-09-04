from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import LabConfig, load_config
from ...control.chunk import ChunkStreamConfig, limit_tcp_target
from ...control.gripper_policy import GripperCommandConfig, GripperPolicy
from ...control.servoj import ServoJController, ServoJStreamConfig
from ...hardware.gripper import GripperSerial
from ...hardware.realsense import DualColorCamera


@dataclass(frozen=True)
class InferenceConfig:
    task_name: str
    task_config: str
    episode_count: int
    checkpoint_name: str
    inference_hz: float = 10.0
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.5
    action_delta_scale: float = 1.0
    close_threshold: float = 0.60
    open_threshold: float = 0.40
    stable_count: int = 3
    minimum_command_interval_s: float = 2.0
    maximum_cycles: int = 1


def _import_act_policy(act_root: Path):
    if not act_root.is_dir():
        raise FileNotFoundError(f"RoboTwin ACT directory not found: {act_root}")
    sys.path.insert(0, str(act_root))
    original_argv = list(sys.argv)
    sys.argv = [
        original_argv[0],
        "--ckpt_dir",
        "bootstrap-only",
        "--policy_class",
        "ACT",
        "--task_name",
        "bootstrap-only",
        "--seed",
        "0",
        "--num_epochs",
        "1",
        "--state_dim",
        "14",
    ]
    try:
        from act_policy import ACTPolicy
    finally:
        sys.argv = original_argv
    return ACTPolicy


def _load_task(act_root: Path, key: str) -> dict:
    path = act_root / "SIM_TASK_CONFIGS.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if key not in config:
        raise KeyError(f"task key not found in {path}: {key}")
    return config[key]


def _load_policy(act_root: Path, inference: InferenceConfig, camera_names: list[str]):
    import torch

    policy_type = _import_act_policy(act_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_config = {
        "lr": 1e-5,
        "num_queries": 50,
        "chunk_size": 50,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
    }
    policy = policy_type(policy_config).to(device)
    policy.eval()
    checkpoint_dir = (
        act_root
        / "act_ckpt"
        / f"act-{inference.task_name}"
        / f"{inference.task_config}-{inference.episode_count}"
    )
    checkpoint_path = checkpoint_dir / inference.checkpoint_name
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    print(f"[MODEL] {policy.load_state_dict(state, strict=False)}")
    with (checkpoint_dir / "dataset_stats.pkl").open("rb") as handle:
        stats = pickle.load(handle)
    return policy, device, stats


def build_qpos(tcp, gripper_value: float):
    import numpy as np

    tcp_array = np.asarray(tcp, dtype=np.float32)
    # Training conversion keeps the unused left gripper at zero and stores the
    # physical single gripper in the right-gripper slot.
    return np.concatenate(
        [tcp_array, np.asarray([0.0], dtype=np.float32), tcp_array, np.asarray([gripper_value], dtype=np.float32)]
    )


def build_images(head, wrist, camera_names: list[str], device: Any):
    import cv2
    import numpy as np
    import torch

    images_by_name = {
        "cam_high": cv2.resize(head, (640, 480)),
        "cam_right_wrist": cv2.resize(wrist, (640, 480)),
        "cam_left_wrist": cv2.resize(wrist, (640, 480)),
    }
    unknown = [name for name in camera_names if name not in images_by_name]
    if unknown:
        raise ValueError(f"unsupported ACT camera names: {unknown}")
    values = np.stack([images_by_name[name] for name in camera_names]).astype(np.float32) / 255.0
    values = np.transpose(values, (0, 3, 1, 2))
    return torch.from_numpy(values).to(device).unsqueeze(0)


def run_inference(
    lab: LabConfig,
    robotwin_root: Path,
    inference: InferenceConfig,
    *,
    execute: bool,
    enable_gripper: bool,
) -> None:
    act_root = robotwin_root / "policy" / "ACT"
    task_key = f"sim-{inference.task_name}-{inference.task_config}-{inference.episode_count}"
    task = _load_task(act_root, task_key)
    if not execute:
        print(f"task: {task_key}")
        print(f"cameras: {task['camera_names']}")
        print("[DRY RUN] add --execute to connect cameras and move the robot")
        return

    import numpy as np
    import torch

    policy, device, stats = _load_policy(act_root, inference, task["camera_names"])
    cameras = DualColorCamera(
        lab.cameras.head_serial,
        lab.cameras.wrist_serial,
        lab.cameras.width,
        lab.cameras.height,
        lab.cameras.fps,
        lab.cameras.warmup_frames,
    )
    controller = ServoJController(
        ServoJStreamConfig(
            robot_host=lab.robot.host,
            robot_port=lab.robot.rtde_port,
            frequency_hz=lab.servoj.frequency_hz,
            config_xml_path=lab.servoj.config_xml,
            servoj_mode=lab.servoj.mode,
            connect_timeout_s=lab.servoj.connect_timeout_s,
        )
    )
    gripper = (
        GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s)
        if enable_gripper
        else None
    )
    gripper_config = GripperCommandConfig(
        close_threshold=inference.close_threshold,
        open_threshold=inference.open_threshold,
        stable_count=inference.stable_count,
        minimum_command_interval_s=inference.minimum_command_interval_s,
        maximum_cycles=inference.maximum_cycles,
    )
    gripper_policy = GripperPolicy(gripper, gripper_config) if gripper is not None else None
    motion_config = ChunkStreamConfig(
        policy_hz=inference.inference_hz,
        servo_hz=lab.servoj.frequency_hz,
        max_linear_velocity=inference.max_linear_velocity,
        max_angular_velocity=inference.max_angular_velocity,
        action_delta_scale=inference.action_delta_scale,
    )
    cameras.start()
    controller.connect_and_start()
    period = 1.0 / inference.inference_hz
    next_step = time.monotonic()
    try:
        while True:
            wait = next_step - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_step += period
            tcp = controller.get_latest_tcp()
            pair = cameras.read()
            if tcp is None or pair is None:
                continue
            estimated_gripper = 0.0 if gripper_policy is None else gripper_policy.estimated
            qpos = build_qpos(tcp, estimated_gripper)
            normalized = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
            qpos_tensor = torch.from_numpy(normalized).float().to(device).unsqueeze(0)
            image_tensor = build_images(pair.head, pair.wrist, task["camera_names"], device)
            with torch.no_grad():
                prediction = policy(qpos_tensor, image_tensor)
            raw = prediction[0, 0].detach().cpu().numpy()
            action = raw * stats["action_std"] + stats["action_mean"]
            target = limit_tcp_target(np.asarray(tcp), action[:6], period, motion_config)
            controller.set_target_tcp(target)
            if gripper_policy is not None:
                gripper_policy.step(float(action[13]), time.monotonic())
    except KeyboardInterrupt:
        print("\n[STOP] inference interrupted")
    finally:
        controller.stop()
        cameras.stop()
        if gripper is not None:
            gripper.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RoboTwin ACT on the real UR5e")
    parser.add_argument("--config", required=True)
    parser.add_argument("--robotwin-root", type=Path, default=Path(".third_party/RoboTwin"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="simple")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--checkpoint", default="policy_best.ckpt")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    run_inference(
        load_config(args.config),
        args.robotwin_root.resolve(),
        InferenceConfig(args.task, args.task_config, args.episodes, args.checkpoint),
        execute=args.execute,
        enable_gripper=not args.no_gripper,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
