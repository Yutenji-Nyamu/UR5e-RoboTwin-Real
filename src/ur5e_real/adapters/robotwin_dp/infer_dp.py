from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import LabConfig, load_config
from ...control.chunk import ChunkStreamConfig, stream_tcp_chunk
from ...control.gripper_policy import GripperCommandConfig, GripperPolicy
from ...control.servoj import ServoJController, ServoJStreamConfig
from ...control.socket_speedl import SocketSpeedLConfig, smoothed_speedl_target
from ...data.schema import nearest_rotation_vector
from ...hardware.dashboard import require_external_motion_ready
from ...hardware.gripper import GripperSerial
from ...hardware.realsense import DualColorCamera, LatestDualColorCamera
from ...hardware.rtde import LatestRtdeTcpClient, RtdeOutputConfig, RtdeTcpClient
from ...hardware.urscript import send_urscript, speedl_command, stopl_command

DP_IMAGE_SIZE = (320, 240)  # width, height


@dataclass(frozen=True)
class InferenceConfig:
    n_obs_steps: int = 3
    n_action_steps: int = 6
    policy_hz: float = 10.0
    chunks: int = 1
    gpu: str = "0"
    enable_gripper: bool = True
    backend: str = "socket"
    smoothing_alpha: float = 0.7


def _image_chw(image, image_size: tuple[int, int] = DP_IMAGE_SIZE):
    import cv2
    import numpy as np

    resized = cv2.resize(image, image_size)
    return np.moveaxis(resized, -1, 0).astype(np.float32) / 255.0


def observation_from_vector(vector, head, wrist) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(vector, dtype=np.float32)
    if values.shape != (14,):
        raise ValueError("DP state vector must contain 14 values")
    return {
        "head_cam": _image_chw(head),
        "left_cam": _image_chw(wrist),
        "right_cam": _image_chw(wrist),
        "agent_pos": values,
    }


class RealObservationEncoder:
    """Encode live TCP observations while keeping the axis-angle branch continuous."""

    def __init__(self) -> None:
        self._last_rotation: list[float] | None = None

    def encode(self, tcp, gripper_value: float, head, wrist) -> dict[str, Any]:
        import numpy as np

        pose = np.asarray(tcp, dtype=np.float32).copy()
        if pose.shape != (6,):
            raise ValueError("TCP pose must contain six values")
        if self._last_rotation is not None:
            pose[3:6] = nearest_rotation_vector(pose[3:6], self._last_rotation)
        self._last_rotation = pose[3:6].tolist()
        vector = np.concatenate((pose, [0.0], pose, [float(gripper_value)])).astype(np.float32)
        return observation_from_vector(vector, head, wrist)


def load_model(robotwin_root: Path, checkpoint: Path, inference: InferenceConfig):
    import yaml

    dp_root = robotwin_root / "policy" / "DP"
    config_path = dp_root / "diffusion_policy" / "config" / "robot_dp_14.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    n_obs_steps = int(model_config["n_obs_steps"])
    n_action_steps = int(model_config["n_action_steps"])
    if (n_obs_steps, n_action_steps) != (inference.n_obs_steps, inference.n_action_steps):
        raise ValueError(
            f"adapter expects {inference.n_obs_steps}/{inference.n_action_steps} observation/action steps, "
            f"checkpoint config uses {n_obs_steps}/{n_action_steps}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = inference.gpu
    sys.path.insert(0, str(dp_root))
    from dp_model import DP

    print(f"[MODEL] {checkpoint}")
    return DP(str(checkpoint), n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def _decode_jpeg(encoded):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode HDF5 image")
    return image


def run_offline(
    robotwin_root: Path,
    checkpoint: Path,
    episode: Path,
    inference: InferenceConfig,
    *,
    index: int = 2,
    output: Path | None = None,
) -> dict[str, Any]:
    import h5py
    import numpy as np

    model = load_model(robotwin_root, checkpoint, inference)
    with h5py.File(episode, "r") as root:
        vectors = root["joint_action/vector"][()]
        head_images = root["observation/head_camera/rgb"][()]
        wrist_images = root["observation/right_camera/rgb"][()]
    length = min(len(vectors), len(head_images), len(wrist_images))
    if not 0 <= index < length - 1:
        raise ValueError(f"index must be in [0, {length - 2}]")

    model.reset_obs()
    history_start = max(0, index - inference.n_obs_steps + 1)
    for frame_index in range(history_start, index + 1):
        model.update_obs(
            observation_from_vector(
                vectors[frame_index],
                _decode_jpeg(head_images[frame_index]),
                _decode_jpeg(wrist_images[frame_index]),
            )
        )
    started = time.perf_counter()
    predicted = np.asarray(model.get_action(), dtype=np.float32)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    target = np.asarray(vectors[index + 1 : index + 1 + len(predicted)], dtype=np.float32)
    compared = predicted[: len(target)]
    metrics = {
        "index": index,
        "predicted_steps": len(predicted),
        "compared_steps": len(target),
        "inference_ms": elapsed_ms,
        "mse": float(np.mean((compared - target) ** 2)),
        "max_abs_error": float(np.max(np.abs(compared - target))),
    }
    print(
        f"[OFFLINE] index={index} chunk={len(predicted)} inference={elapsed_ms:.1f}ms "
        f"mse={metrics['mse']:.6f} max_abs={metrics['max_abs_error']:.6f}"
    )
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, predicted=predicted, target=target, index=index)
        print(f"[SAVED] {output}")
    return {**metrics, "predicted": predicted, "target": target}


def _camera(lab: LabConfig) -> DualColorCamera:
    return DualColorCamera(
        lab.cameras.head_serial,
        lab.cameras.wrist_serial,
        lab.cameras.width,
        lab.cameras.height,
        lab.cameras.fps,
        lab.cameras.warmup_frames,
    )


def _servo(lab: LabConfig) -> ServoJController:
    return ServoJController(
        ServoJStreamConfig(
            robot_host=lab.robot.host,
            robot_port=lab.robot.rtde_port,
            config_xml_path=lab.servoj.config_xml,
            frequency_hz=lab.servoj.frequency_hz,
            servoj_mode=lab.servoj.mode,
            connect_timeout_s=lab.servoj.connect_timeout_s,
        )
    )


def _start_servoj_program(lab: LabConfig) -> None:
    require_external_motion_ready(lab.robot.host)
    script = lab.servoj.program_script
    if not script.is_file():
        raise FileNotFoundError(script)
    send_urscript(
        script.read_text(encoding="utf-8"),
        lab.robot.host,
        lab.robot.script_port,
        lab.robot.socket_timeout_s,
    )
    print(f"[CONTROL] robot-side servoJ program started: {script.name}")


def run_shadow(
    lab: LabConfig,
    robotwin_root: Path,
    checkpoint: Path,
    inference: InferenceConfig,
) -> None:
    import numpy as np

    model = load_model(robotwin_root, checkpoint, inference)
    encoder = RealObservationEncoder()
    cameras = LatestDualColorCamera(_camera(lab))
    states = LatestRtdeTcpClient(
        RtdeTcpClient(RtdeOutputConfig(lab.robot.host, lab.robot.rtde_port, inference.policy_hz))
    )
    cameras.start()
    states.start()
    model.reset_obs()
    print("[SHADOW] running predictions only; no robot or gripper commands")
    chunk_index = 0
    try:
        while inference.chunks == 0 or chunk_index < inference.chunks:
            state = states.read()
            pair = cameras.read()
            if state is None or pair is None:
                continue
            observation = encoder.encode(state[1], 0.0, pair.head, pair.wrist)
            started = time.perf_counter()
            actions = np.asarray(model.get_action(observation), dtype=np.float32)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(
                f"[CHUNK {chunk_index}] {len(actions)} actions, {elapsed_ms:.1f}ms, "
                f"first_tcp={np.round(actions[0, :6], 4).tolist()} gripper={actions[0, 13]:.3f}"
            )
            deadline = time.monotonic()
            for _ in actions:
                deadline += 1.0 / inference.policy_hz
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                state = states.read()
                pair = cameras.read()
                if state is not None and pair is not None:
                    model.update_obs(encoder.encode(state[1], 0.0, pair.head, pair.wrist))
            chunk_index += 1
    except KeyboardInterrupt:
        print("\n[STOP] shadow inference interrupted")
    finally:
        states.stop()
        cameras.stop()


def _run_execute_rtde(
    lab: LabConfig,
    robotwin_root: Path,
    checkpoint: Path,
    inference: InferenceConfig,
) -> None:
    import numpy as np

    model = load_model(robotwin_root, checkpoint, inference)
    encoder = RealObservationEncoder()
    cameras = LatestDualColorCamera(_camera(lab))
    controller = _servo(lab)
    gripper = (
        GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s)
        if inference.enable_gripper
        else None
    )
    gripper_policy = GripperPolicy(gripper, GripperCommandConfig()) if gripper is not None else None
    stream_config = ChunkStreamConfig(policy_hz=inference.policy_hz, servo_hz=lab.servoj.frequency_hz)
    chunk_index = 0
    try:
        cameras.start()
        _start_servoj_program(lab)
        controller.connect_and_start()
        model.reset_obs()
        print("[EXECUTE] continuous 6-action chunks through RTDE servoJ")
        while inference.chunks == 0 or chunk_index < inference.chunks:
            tcp = controller.get_latest_tcp()
            pair = cameras.read()
            if tcp is None or pair is None:
                continue
            gripper_value = 0.0 if gripper_policy is None else gripper_policy.estimated
            observation = encoder.encode(tcp, gripper_value, pair.head, pair.wrist)
            started = time.perf_counter()
            actions = np.asarray(model.get_action(observation), dtype=np.float32)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(f"[CHUNK {chunk_index}] {len(actions)} actions, inference={elapsed_ms:.1f}ms")

            captured_observations = []

            def update_observation(action_index: int, _target: list[float]) -> None:
                action = actions[action_index]
                if gripper_policy is not None:
                    gripper_policy.step(float(action[13]))
                tcp = controller.get_latest_tcp()
                pair = cameras.read()
                if tcp is not None and pair is not None:
                    gripper_value = 0.0 if gripper_policy is None else gripper_policy.estimated
                    captured_observations.append((tcp, gripper_value, pair))

            stream_tcp_chunk(controller, actions[:, :6], stream_config, on_waypoint=update_observation)
            for tcp, gripper_value, pair in captured_observations:
                model.update_obs(encoder.encode(tcp, gripper_value, pair.head, pair.wrist))
            chunk_index += 1
    except KeyboardInterrupt:
        print("\n[STOP] DP execution interrupted")
    finally:
        controller.stop()
        cameras.stop()
        if gripper is not None:
            gripper.shutdown()


def _run_execute_socket(
    lab: LabConfig,
    robotwin_root: Path,
    checkpoint: Path,
    inference: InferenceConfig,
) -> None:
    import numpy as np

    model = load_model(robotwin_root, checkpoint, inference)
    encoder = RealObservationEncoder()
    cameras = LatestDualColorCamera(_camera(lab))
    states = LatestRtdeTcpClient(
        RtdeTcpClient(RtdeOutputConfig(lab.robot.host, lab.robot.rtde_port, inference.policy_hz))
    )
    gripper = (
        GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s)
        if inference.enable_gripper
        else None
    )
    gripper_policy = GripperPolicy(gripper, GripperCommandConfig()) if gripper is not None else None
    motion = SocketSpeedLConfig(
        policy_hz=inference.policy_hz,
        smoothing_alpha=inference.smoothing_alpha,
    )
    robot: socket.socket | None = None
    previous_target = None
    chunk_index = 0
    try:
        cameras.start()
        states.start()
        require_external_motion_ready(lab.robot.host)
        robot = socket.create_connection(
            (lab.robot.host, lab.robot.script_port), timeout=lab.robot.socket_timeout_s
        )
        model.reset_obs()
        print(
            f"[EXECUTE] socket speedL at {inference.policy_hz:g} Hz; "
            f"target EMA alpha={inference.smoothing_alpha:g}"
        )
        while inference.chunks == 0 or chunk_index < inference.chunks:
            state = states.read()
            pair = cameras.read()
            if state is None or pair is None:
                continue
            tcp = state[1]
            gripper_value = 0.0 if gripper_policy is None else gripper_policy.estimated
            observation = encoder.encode(tcp, gripper_value, pair.head, pair.wrist)
            started = time.perf_counter()
            actions = np.asarray(model.get_action(observation), dtype=np.float32)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(f"[CHUNK {chunk_index}] {len(actions)} actions, inference={elapsed_ms:.1f}ms")

            deadline = time.monotonic()
            captured_observations = []
            for action in actions:
                state = states.read()
                if state is None:
                    continue
                tcp = state[1]
                previous_target, velocity = smoothed_speedl_target(
                    tcp, action[:6], previous_target, motion
                )
                robot.sendall(
                    speedl_command(
                        velocity,
                        acceleration=motion.acceleration,
                        duration_s=1.0 / motion.policy_hz,
                    ).encode("utf-8")
                )
                if gripper_policy is not None:
                    gripper_policy.step(float(action[13]))
                deadline += 1.0 / motion.policy_hz
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                latest_state = states.read()
                pair = cameras.read()
                if latest_state is not None and pair is not None:
                    gripper_value = 0.0 if gripper_policy is None else gripper_policy.estimated
                    captured_observations.append((latest_state[1], gripper_value, pair))
            for tcp, gripper_value, pair in captured_observations:
                model.update_obs(encoder.encode(tcp, gripper_value, pair.head, pair.wrist))
            chunk_index += 1
    except KeyboardInterrupt:
        print("\n[STOP] DP execution interrupted")
    finally:
        if robot is not None:
            try:
                robot.sendall(stopl_command().encode("utf-8"))
            except Exception:
                pass
            robot.close()
        states.stop()
        cameras.stop()
        if gripper is not None:
            gripper.shutdown()


def run_execute(
    lab: LabConfig,
    robotwin_root: Path,
    checkpoint: Path,
    inference: InferenceConfig,
) -> None:
    if inference.backend == "socket":
        _run_execute_socket(lab, robotwin_root, checkpoint, inference)
    elif inference.backend == "rtde":
        _run_execute_rtde(lab, robotwin_root, checkpoint, inference)
    else:
        raise ValueError(f"unsupported execution backend: {inference.backend}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RoboTwin DP offline or on the real UR5e")
    parser.add_argument("--robotwin-root", type=Path, default=Path(".third_party/RoboTwin"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--episode", type=Path, help="offline HDF5 episode; never connects hardware")
    mode.add_argument("--shadow", action="store_true", help="live inference without sending commands")
    mode.add_argument("--execute", action="store_true", help="execute through the selected motion backend")
    parser.add_argument("--config", help="lab config required for shadow or execute")
    parser.add_argument("--index", type=int, default=2, help="offline observation index")
    parser.add_argument("--output", type=Path, help="optional offline prediction NPZ")
    parser.add_argument("--chunks", type=int, default=1, help="live chunks; 0 means until Ctrl+C")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--backend", choices=("socket", "rtde"), default="socket")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="socket target EMA; 1 disables smoothing (default: 0.7)",
    )
    args = parser.parse_args(argv)
    inference = InferenceConfig(
        chunks=args.chunks,
        gpu=args.gpu,
        enable_gripper=not args.no_gripper,
        backend=args.backend,
        smoothing_alpha=args.smooth_alpha,
    )
    robotwin_root = args.robotwin_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.episode is not None:
        run_offline(
            robotwin_root,
            checkpoint,
            args.episode.expanduser().resolve(),
            inference,
            index=args.index,
            output=args.output,
        )
        return 0
    if not args.config:
        parser.error("--config is required with --shadow or --execute")
    lab = load_config(args.config)
    if args.shadow:
        run_shadow(lab, robotwin_root, checkpoint, inference)
    else:
        run_execute(lab, robotwin_root, checkpoint, inference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
