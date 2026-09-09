"""Offline, read-only shadow, or explicitly requested joint execution."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from ...config import load_config
from ...control.gripper_policy import GripperCommandConfig, GripperPolicy
from ...control.joint import plan_joint_chunk, stream_joint_chunk
from ...control.joint_servoj import JointServoJController
from ...hardware.gripper import GripperSerial
from ...hardware.rtde import RtdeOutputConfig, RtdeStateClient
from .client import PolicyClient
from .contract import decode_actions, motion_config, require_home
from .dataset import offline_observation, validate_dataset, write_json
from .runtime import FreshCameras, controller_config, observation

DEFAULT_ACTION_STEPS = 20


def run(args):
    contract, _ = validate_dataset(args.dataset)
    if not 1 <= args.action_steps <= 50 or args.chunks < 1:
        raise ValueError("action steps must be 1..50 and chunks must be bounded and positive")
    if not 0 < args.speed <= 0.6:
        raise ValueError("this demo supports a joint software speed limit in (0, 0.6] rad/s")
    with ExitStack() as stack:
        client = stack.enter_context(PolicyClient(contract, port=args.port, timeout_s=args.timeout))
        expected_instance = getattr(args, "expected_instance_id", None)
        if expected_instance is not None and client.metadata.get("instance_id") != expected_instance:
            raise RuntimeError("model service instance changed; refuse hardware access")
        if args.mode == "offline":
            obs, expected = offline_observation(args.dataset, args.run_id, args.index)
            result = client.infer(obs)
            joints, _ = decode_actions(result["actions"])
            metrics = {
                "mode": "offline",
                "client_elapsed_s": result["client_elapsed_s"],
                "joint_mae_rad": float(np.mean(np.abs(joints - expected[:, :6]))),
                "joint_max_error_rad": float(np.max(np.abs(joints - expected[:, :6]))),
                "predicted_actions": np.asarray(result["actions"]).tolist(),
                "physical_execution": False,
            }
            if args.output:
                write_json(args.output, metrics)
            print(json.dumps(metrics, indent=2))
            return
        if args.lab_config is None:
            raise ValueError("live modes require --lab-config")
        if args.mode == "execute" and client.metadata.get("training_status") != "SFT_not_physical_validation":
            raise ValueError("development smoke checkpoints are offline/shadow only; train a demo checkpoint first")
        lab = load_config(args.lab_config)
        cameras = stack.enter_context(FreshCameras(lab.cameras))
        reader = stack.enter_context(RtdeStateClient(RtdeOutputConfig(lab.robot.host, lab.robot.rtde_port, 10)))
        state = reader.receive_state()
        if state is None:
            raise RuntimeError("missing joint state")
        pair = cameras.read()
        # Live-shape warmup before creating any motion controller or gripper serial connection.
        client.infer(observation(state.actual_q, 0.0, pair.head, pair.wrist, contract["prompt"]))
        gripper_policy = None
        controller = None
        if args.mode == "execute":
            require_home(state.actual_q, contract)
            reader.close()
            controller = JointServoJController(controller_config(lab, contract, speed=args.speed))
            stack.callback(controller.stop)
            gripper = stack.enter_context(GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s))
            gripper.serial.write_timeout = 0.1
            gripper.open()
            time.sleep(1.0)
            controller.connect_and_prime()
            require_home(controller.get_latest_joints(), contract)
            controller.start()
            # Use the existing sparse one-close/one-open policy, with explicit MVP timing.
            gripper_policy = GripperPolicy(
                gripper, GripperCommandConfig(stable_count=2, minimum_command_interval_s=0.5, maximum_cycles=1)
            )
        motion = motion_config(contract, speed=args.speed)
        reports = []
        for chunk in range(args.chunks):
            state = controller.get_latest_state() if controller else reader.receive_state()
            if state is None:
                raise RuntimeError("RTDE state stream closed")
            pair = cameras.read()
            grip = gripper_policy.estimated if gripper_policy else args.shadow_gripper
            obs = observation(state.actual_q, grip, pair.head, pair.wrist, contract["prompt"])
            result = client.infer(obs)
            joints, grips = decode_actions(result["actions"])
            targets = joints[: args.action_steps]
            report = {
                "chunk": chunk + 1,
                "mode": args.mode,
                "client_elapsed_s": result["client_elapsed_s"],
                "action_horizon": contract["action_horizon"],
                "action_steps": args.action_steps,
                "policy_hz": motion.policy_hz,
            }
            if controller:
                # Preflight every executed target before sending the first servo command.
                plan_joint_chunk(controller.get_commanded_joints(), targets, motion)
                report.update(
                    stream_joint_chunk(
                        controller,
                        targets,
                        motion,
                        on_waypoint=lambda i, _q: gripper_policy.step(float(grips[i])),
                        finish_after_waypoint=lambda _i: gripper_policy.cycles > 0,
                    )
                )
            else:
                try:
                    plan_joint_chunk(state.actual_q, targets, motion)
                    report["execution_preflight"] = "pass"
                except ValueError as exc:
                    report["execution_preflight"] = str(exc)
            print(json.dumps(report), flush=True)
            reports.append(report)
            if gripper_policy and gripper_policy.cycles:
                # Stop new policy motion after the release; retain one second of stable hold.
                hold = np.asarray(controller.get_commanded_joints())
                stream_joint_chunk(controller, np.repeat(hold[None], 10, axis=0), motion)
                break
        if args.output:
            write_json(args.output, reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--mode", choices=("offline", "shadow", "execute"), default="offline")
    parser.add_argument("--lab-config", type=Path)
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--timeout", type=float, default=1.5)
    parser.add_argument(
        "--action-steps",
        type=int,
        default=DEFAULT_ACTION_STEPS,
        help="executed prefix K (default: 20, user-selected UR trial); model horizon H remains 50",
    )
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument(
        "--shadow-gripper",
        type=float,
        choices=(0.0, 1.0),
        default=0.0,
        help="assumed commanded state; shadow never opens gripper serial",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
