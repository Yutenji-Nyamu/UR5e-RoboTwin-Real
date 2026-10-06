"""Single physical arm, fresh RGB, bounded chunks; no model framework in this process."""

from contextlib import ExitStack
import json
import time

import numpy as np

from ...config import load_config
from ...control.gripper_policy import GripperCommandConfig, GripperPolicy
from ...control.joint import joint_vector, plan_joint_chunk, stream_joint_chunk
from ...control.joint_servoj import JointServoJController
from ...hardware.gripper import GripperSerial
from ...hardware.rtde import RtdeOutputConfig, RtdeStateClient
from ..robotwin_pi05.runtime import FreshCameras, controller_config
from .client import PolicyClient
from .contract import decode_actions, motion_config, require_home
from .data import image_paths, read_dataset, read_rgb


def observation(q, gripper, head_bgr, wrist_bgr):
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("commanded gripper state must be in [0, 1]")
    result = {"state": np.asarray([*joint_vector(q), gripper], dtype=np.float32)}
    for key, view in (("head", head_bgr), ("wrist", wrist_bgr)):
        if view.ndim != 3 or view.shape[-1] != 3 or view.dtype != np.uint8:
            raise ValueError("live camera images must be HWC uint8 BGR")
        result[key] = view[..., ::-1].copy()
    return result


def home(args):
    contract, *_ = read_dataset(args.dataset)
    target = np.asarray(contract["home_q"])
    command = "close" if contract["initial_gripper"] else "open"
    report = {"home_q_rad": target.tolist(), "gripper": command, "max_speed_rad_s": 0.1,
              "physical_execution": args.execute}
    if not args.execute:
        return report
    if args.lab_config is None:
        raise ValueError("home --execute requires --lab-config")
    print(json.dumps(report), flush=True)
    lab = load_config(args.lab_config)
    config = controller_config(lab, contract, speed=0.1)
    with JointServoJController(config) as controller:
        current = np.asarray(controller.get_latest_joints())
        steps = max(1, int(np.ceil(np.max(np.abs(target - current)) / 0.01)))
        if steps > 300:
            raise ValueError("home path exceeds 30s; return near the demo start manually")
        targets = current + np.arange(1, steps + 1)[:, None] / steps * (target - current)
        stream_joint_chunk(controller, targets, config.motion)
        require_home(controller.get_latest_joints(), contract)
    with GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s) as gripper:
        getattr(gripper, command)()
        time.sleep(1)
    return report


def run(args):
    contract, states, actions, valid, images = read_dataset(args.dataset)
    if not 1 <= args.action_steps <= 50 or args.chunks < 1 or not 0 < args.speed <= 0.6:
        raise ValueError("action steps must be 1..50, chunks positive, joint speed in (0, 0.6] rad/s")
    reports = []
    with ExitStack() as stack:
        client = stack.enter_context(PolicyClient(contract, port=args.port, timeout_s=args.timeout))
        if args.expected_instance_id and client.metadata.get("instance_id") != args.expected_instance_id:
            raise RuntimeError("model service instance changed")
        if args.mode == "offline":
            if not 0 <= args.index < len(states):
                raise ValueError("offline index outside the dataset")
            head, wrist = image_paths(args.dataset, images[args.index], data_root=args.data_root)
            result = client.infer({"state": states[args.index], "head": read_rgb(head), "wrist": read_rgb(wrist)})
            predicted = result["actions"]
            keep = valid[args.index]
            report = {"mode": "offline", "run_id": images[args.index]["run_id"],
                      "split": images[args.index]["split"], "physical_execution": False,
                      "client_elapsed_s": result["client_elapsed_s"], "valid_steps": int(keep.sum()),
                      "joint_mae_rad": float(np.abs(predicted[keep, :6] - actions[args.index, keep, :6]).mean()),
                      "gripper_mae": float(np.abs(predicted[keep, 6] - actions[args.index, keep, 6]).mean())}
            print(json.dumps(report), flush=True)
            reports.append(dict(report, actions=predicted.tolist()))
        else:
            if args.lab_config is None:
                raise ValueError("shadow/execute require --lab-config")
            if args.mode == "execute" and client.metadata.get("training_status") != "SFT_not_physical_validation":
                raise ValueError("execute requires a trained full-size checkpoint; diagnostics cannot drive hardware")
            lab = load_config(args.lab_config)
            cameras = stack.enter_context(FreshCameras(lab.cameras))
            reader = stack.enter_context(RtdeStateClient(RtdeOutputConfig(lab.robot.host, lab.robot.rtde_port, 10)))
            state = reader.receive_state()
            if state is None:
                raise RuntimeError("missing joint state")
            pair = cameras.read()
            initial = contract["initial_gripper"] if args.shadow_gripper is None else args.shadow_gripper
            if args.mode == "execute":
                initial = contract["initial_gripper"]
            # A full live-shape request must pass before motion or gripper connections exist.
            client.infer(observation(state.actual_q, initial, pair.head, pair.wrist))
            controller = gripper_policy = None
            if args.mode == "execute":
                require_home(state.actual_q, contract)
                reader.close()
                controller = JointServoJController(controller_config(lab, contract, speed=args.speed))
                stack.callback(controller.stop)
                gripper = stack.enter_context(GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s))
                gripper.serial.write_timeout = 0.1
                getattr(gripper, "close" if initial else "open")()
                time.sleep(1)
                controller.connect_and_prime()
                require_home(controller.get_latest_joints(), contract)
                controller.start()
                gripper_policy = GripperPolicy(gripper, GripperCommandConfig(
                    stable_count=2, minimum_command_interval_s=0.5, maximum_cycles=None), initial_state=initial)
            motion = motion_config(contract, speed=args.speed)
            for chunk in range(args.chunks):
                state = controller.get_latest_state() if controller else reader.receive_state()
                if state is None:
                    raise RuntimeError("RTDE state stream closed")
                pair = cameras.read()
                grip = gripper_policy.estimated if gripper_policy else initial
                result = client.infer(observation(state.actual_q, grip, pair.head, pair.wrist))
                joints, grips = decode_actions(result["actions"])
                targets = joints[:args.action_steps]
                report = {"mode": args.mode, "chunk": chunk + 1, "client_elapsed_s": result["client_elapsed_s"],
                          "action_steps": args.action_steps, "physical_execution": controller is not None}
                if controller:
                    plan_joint_chunk(controller.get_commanded_joints(), targets, motion)
                    report.update(stream_joint_chunk(controller, targets, motion,
                                  on_waypoint=lambda i, _q: gripper_policy.step(float(grips[i]))))
                    report["gripper_cycles"] = gripper_policy.cycles
                else:
                    try:
                        plan_joint_chunk(state.actual_q, targets, motion)
                        report["execution_preflight"] = "pass"
                    except ValueError as exc:
                        report["execution_preflight"] = str(exc)
                print(json.dumps(report), flush=True)
                reports.append(report)
            # No automatic release or stop-on-open: final close remains meaningful.
    if args.output:
        with args.output.open("x") as handle:
            json.dump(reports, handle, indent=2)
            handle.write("\n")
    return {"mode": args.mode, "reports": len(reports), "physical_execution": args.mode == "execute"}
