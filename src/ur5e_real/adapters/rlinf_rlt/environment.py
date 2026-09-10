"""UR5 EnvWorker boundary: measured observations, servoJ actions and terminal results.

Importing or constructing this module never opens a device. reset/start are only
called by the explicitly requested execute round after the operator's prompts.
"""

from __future__ import annotations

from contextlib import ExitStack
import time
import numpy as np


def executor_aux(policy, now=None):
    now = time.monotonic() if now is None else now
    phase = [float(policy.state == value) for value in ("wait_close", "wait_open", "done")]
    interval = policy.config.minimum_command_interval_s
    return np.asarray(
        [
            *phase,
            min(1.0, policy.close_streak / policy.config.stable_count),
            min(1.0, policy.open_streak / policy.config.stable_count),
            max(0.0, min(1.0, (interval - now + policy.last_command_at) / interval)),
        ],
        dtype=np.float32,
    )


class RealEnvironment:
    def __init__(self, run):
        self.run, self.cfg, self.contract = run, run["config"]["real"], run["contract"]
        self.stack = None
        self.last_trace = {}

    def reset(self):
        from ...config import load_config
        from ..robotwin_pi05.runtime import prepare

        self.close()
        self.lab = load_config(self.run["lab_config"])
        prepare(self.lab, self.contract, execute=True)

    def start(self):
        from ...control.gripper_policy import GripperPolicy, GripperCommandConfig
        from ...control.joint_servoj import JointServoJController
        from ...hardware.gripper import GripperSerial
        from ..robotwin_pi05.contract import require_home, motion_config
        from ..robotwin_pi05.runtime import FreshCameras, controller_config

        self.stack = ExitStack()
        try:
            self.cameras = self.stack.enter_context(FreshCameras(self.lab.cameras))
            gripper = self.stack.enter_context(
                GripperSerial(self.lab.gripper.port, self.lab.gripper.baudrate, self.lab.gripper.timeout_s)
            )
            gripper.serial.write_timeout = 0.1
            self.gripper = GripperPolicy(
                gripper, GripperCommandConfig(stable_count=2, minimum_command_interval_s=0.5, maximum_cycles=1)
            )
            self.controller = JointServoJController(
                controller_config(self.lab, self.contract, speed=self.cfg["speed_rad_s"])
            )
            self.stack.callback(self.controller.stop)
            self.controller.connect_and_prime()
            require_home(self.controller.get_latest_joints(), self.contract)
            self.controller.start()
            self.motion = motion_config(self.contract, speed=self.cfg["speed_rad_s"])
            return self.observe()
        except BaseException:
            self.close()
            raise

    def observe(self):
        from ..robotwin_pi05.runtime import observation

        state = self.controller.get_latest_state()
        pair = self.cameras.read()
        obs = observation(state.actual_q, self.gripper.estimated, pair.head, pair.wrist, self.contract["prompt"])
        obs["executor_aux"] = executor_aux(self.gripper)
        obs["rlt_observed_at"] = time.monotonic()
        obs["rlt_controller_time_s"] = state.controller_time_s
        obs["rlt_state_host_time_s"] = state.host_receive_time_s
        return obs

    def step(self, response, *, cancelled):
        from ...control.joint import stream_joint_chunk
        from ..robotwin_pi05.contract import decode_actions

        joints, grips = decode_actions(response["actions"])
        targets = joints[: self.cfg["action_steps"]]
        commands, measured, events = [], [], []
        self.last_trace = {
            "command_t_q": commands,
            "waypoint_t_q_tcp_grip": measured,
            "gripper_events_t_state": events,
            "requested_q": targets,
            "requested_grip": grips[: len(targets)],
        }
        controller, gripper = self.controller, self.gripper

        class TraceController:
            def __getattr__(self, key):
                return getattr(controller, key)

            def set_target_joints(self, q):
                controller.set_target_joints(q)
                commands.append([time.monotonic(), *q])

        def waypoint(i, _q):
            before = gripper.last_command_at
            gripper.step(float(grips[i]))
            state = controller.get_latest_state()
            measured.append([time.monotonic(), *state.actual_q, *state.tcp_pose, gripper.estimated])
            if before != gripper.last_command_at:
                events.append([gripper.last_command_at, gripper.estimated])

        result = stream_joint_chunk(
            TraceController(),
            targets,
            self.motion,
            on_waypoint=waypoint,
            finish_after_waypoint=lambda _i: gripper.cycles > 0,
            cancelled=cancelled,
        )
        released = gripper.cycles > 0
        if released:
            hold = np.asarray(controller.get_commanded_joints())
            count = int(np.ceil(self.cfg["release_hold_s"] * self.motion.policy_hz))
            stream_joint_chunk(
                TraceController(), np.repeat(hold[None], count, axis=0), self.motion, cancelled=cancelled
            )
        self.last_trace["release_hold_s"] = np.asarray(self.cfg["release_hold_s"] if released else 0.0)
        return self.observe(), int(result["executed_waypoints"]), released, self.last_trace

    def close(self):
        if self.stack is not None:
            self.stack.close()
            self.stack = None
