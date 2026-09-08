"""Joint observation/execution boundary; never reinterpret TCP values as joints."""

from __future__ import annotations

import threading
import time

import numpy as np

from ...control.joint import stream_joint_chunk
from ...control.joint_servoj import JointServoJConfig, JointServoJController
from ...hardware.realsense import DualColorCamera
from .contract import encode_state, motion_config, require_home
from .native import REPOSITORY


def observation(q, gripper, head_bgr, wrist_bgr, prompt):
    import cv2

    images = {}
    for key, value in (("cam_high", head_bgr), ("cam_right_wrist", wrist_bgr)):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("live images must be HWC uint8 BGR from the camera adapter")
        images[key] = np.moveaxis(cv2.cvtColor(value, cv2.COLOR_BGR2RGB), -1, 0)
    return {"state": encode_state(q, gripper), "images": images, "prompt": prompt}


class FreshCameras:
    """Timestamp completed frame pairs; fail closed on frozen camera delivery."""

    def __init__(self, camera_config):
        self.camera = DualColorCamera(
            camera_config.head_serial,
            camera_config.wrist_serial,
            camera_config.width,
            camera_config.height,
            camera_config.fps,
            camera_config.warmup_frames,
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._pair = None
        self._at = 0.0
        self._error = None
        self._thread = None

    def _loop(self):
        try:
            while not self._stop.is_set():
                pair = self.camera.read()
                if pair is None:
                    continue
                with self._lock:
                    self._pair, self._at = pair, time.monotonic()
                self._ready.set()
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def read(self, max_age_s=0.2):
        if self._error:
            raise RuntimeError("camera acquisition failed") from self._error
        with self._lock:
            if self._pair is None or time.monotonic() - self._at > max_age_s:
                raise TimeoutError("camera pair is absent or stale")
            return self._pair

    def __enter__(self):
        try:
            self.camera.start()
            self._thread = threading.Thread(target=self._loop, name="pi05-cameras", daemon=True)
            self._thread.start()
            if not self._ready.wait(2):
                raise TimeoutError("no first camera pair")
            self.read()
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        self._stop.set()
        self.camera.stop()
        if self._thread:
            self._thread.join(0.5)


def controller_config(lab, contract, *, speed=0.6):
    return JointServoJConfig(
        robot_host=lab.robot.host,
        robot_port=lab.robot.rtde_port,
        script_port=lab.robot.script_port,
        config_xml=REPOSITORY / "robot_programs" / "joint_control_configuration.xml",
        program_script=REPOSITORY / "robot_programs" / "servoj_joint_control_loop.script",
        motion=motion_config(contract, speed=speed),
        tcp_lower=tuple(contract["tcp_lower"]),
        tcp_upper=tuple(contract["tcp_upper"]),
        tcp_offset=tuple(contract["tcp_offset"]),
    )


def prepare(lab, contract, *, execute=False):
    """Explicit, slow joint-only home. The operator must first clear the direct path."""
    target = np.asarray(contract["home_q"])
    print(f"[JOINT HOME] q={target.tolist()} rad; maximum speed 0.10 rad/s; then open gripper")
    if not execute:
        print("[DRY RUN] no hardware connection; inspect the path before adding --execute")
        return
    from ...hardware.gripper import GripperSerial

    config = controller_config(lab, contract, speed=0.1)
    with JointServoJController(config) as controller:
        current = np.asarray(controller.get_latest_joints())
        steps = max(1, int(np.ceil(np.max(np.abs(target - current)) / 0.01)))
        if steps > 300:
            raise ValueError("home path exceeds 30 seconds; manually return inside the demo starting area")
        targets = current + np.arange(1, steps + 1)[:, None] / steps * (target - current)
        stream_joint_chunk(controller, targets, config.motion)
        require_home(controller.get_latest_joints(), contract)
    with GripperSerial(lab.gripper.port, lab.gripper.baudrate, lab.gripper.timeout_s) as gripper:
        gripper.open()
        time.sleep(1.0)
    print("[READY] measured joint home reached; gripper open command sent")
