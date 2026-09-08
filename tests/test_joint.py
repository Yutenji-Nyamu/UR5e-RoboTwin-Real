from dataclasses import replace
from types import SimpleNamespace
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ur5e_real.control.joint import JointMotionConfig, plan_joint_chunk, stream_joint_chunk
from ur5e_real.control.joint_servoj import JointServoJConfig, JointServoJController, validate_program_ack
from ur5e_real.adapters.robotwin_pi05.native import REPOSITORY


def motion():
    return JointMotionConfig((-4.0,) * 6, (4.0,) * 6)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeController:
    action_space = "joint_position"

    def __init__(self):
        self.q = np.zeros(6)
        self.commands = []
        self.stopped = False

    def get_latest_joints(self):
        return self.q.copy()

    def get_commanded_joints(self):
        return self.q.copy()

    def set_target_joints(self, q):
        self.q = np.array(q)
        self.commands.append(self.q)

    def stop(self):
        self.stopped = True


def test_joint_interpolation_never_wraps_and_preserves_endpoints():
    config = motion()
    start = np.full(6, 3.2)
    targets = np.array([start + 0.03, start + 0.10])
    segments = plan_joint_chunk(start, targets, config)
    assert [len(s) for s in segments] == [50, 59]
    np.testing.assert_allclose([s[-1] for s in segments], targets)
    values = np.concatenate([start[None], *segments])
    assert np.max(np.abs(np.diff(values, axis=0))) * 500 <= 0.6 + 1e-8
    assert np.all(values > np.pi)


@pytest.mark.parametrize("bad", [np.full((2, 6), np.nan), np.zeros((2, 7)), np.full((1, 6), 5), np.full((1, 6), 0.13)])
def test_bad_entire_chunk_rejected_before_any_command(bad):
    controller, clock = FakeController(), FakeClock()
    with pytest.raises(ValueError):
        stream_joint_chunk(controller, bad, motion(), clock=clock, sleep=clock.sleep)
    assert not controller.commands and controller.stopped


def test_waypoint_callback_and_no_catchup_after_short_callback_delay():
    controller, clock = FakeController(), FakeClock()
    reached = []

    def callback(i, q):
        np.testing.assert_allclose(controller.q, q)
        reached.append(i)
        clock.sleep(0.04)

    result = stream_joint_chunk(
        controller, np.full((2, 6), 0.02), motion(), on_waypoint=callback, clock=clock, sleep=clock.sleep
    )
    assert reached == [0, 1] and result["servo_steps"] == 100
    assert clock.now == pytest.approx(0.28)


def test_cancel_tracking_and_lateness_stop_without_gripper():
    for mode in ("cancel", "tracking", "late"):
        controller, clock, callback = FakeController(), FakeClock(), MagicMock()
        if mode == "tracking":
            controller.get_latest_joints = lambda: np.full(6, -0.1)
        if mode == "late":

            def delayed(_seconds):
                clock.now += 0.2

            sleep = delayed
        else:
            sleep = clock.sleep
        with pytest.raises((InterruptedError, RuntimeError, TimeoutError)):
            stream_joint_chunk(
                controller,
                np.full((1, 6), 0.01),
                motion(),
                cancelled=lambda: mode == "cancel",
                on_waypoint=callback,
                clock=clock,
                sleep=sleep,
            )
        assert controller.stopped
        callback.assert_not_called()


def test_chunk_boundary_continues_command_and_release_finishes_without_extra_motion():
    controller, clock = FakeController(), FakeClock()
    controller.q = np.full(6, 0.04)
    controller.get_latest_joints = lambda: controller.q - 0.01
    result = stream_joint_chunk(
        controller,
        np.array([[0.05] * 6, [0.10] * 6]),
        motion(),
        finish_after_waypoint=lambda _i: True,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result["executed_waypoints"] == 1 and not controller.stopped
    assert controller.commands[0][0] > 0.04
    assert controller.commands[-1][0] == pytest.approx(0.05)


def servo_config():
    return JointServoJConfig(
        "robot.test",
        REPOSITORY / "robot_programs/joint_control_configuration.xml",
        REPOSITORY / "robot_programs/servoj_joint_control_loop.script",
        motion(),
        (-1.0,) * 3,
        (1.0,) * 3,
        (0.0,) * 6,
    )


def packet(timestamp=1.0, runtime=1):
    return SimpleNamespace(
        timestamp=timestamp,
        actual_q=[0.2] * 6,
        actual_qd=[0.0] * 6,
        actual_TCP_pose=[0.4] * 6,
        tcp_offset=[0.0] * 6,
        runtime_state=runtime,
        output_int_register_0=10505,
        output_int_register_1=2,
        output_int_register_2=7,
    )


def fake_connection():
    conn = MagicMock()
    conn.receive.return_value = packet()
    conn.send_input_setup.side_effect = [SimpleNamespace(), SimpleNamespace()]
    return conn


def fake_imports(conn):
    recipes = MagicMock()
    recipes.get_recipe.return_value = (["sample"], ["DOUBLE"])
    return SimpleNamespace(RTDE=lambda *_: conn), SimpleNamespace(ConfigFile=lambda *_: recipes)


def test_prime_requires_stopped_robot_then_measured_q_and_zero_mode():
    controller, conn = JointServoJController(servo_config()), fake_connection()
    sent = []
    conn.send.side_effect = lambda obj: sent.append(vars(obj).copy()) or True
    with (
        patch("ur5e_real.control.joint_servoj.require_external_motion_ready") as ready,
        patch("ur5e_real.control.joint_servoj._imports") as imports,
    ):
        imports.return_value = fake_imports(conn)
        controller.connect_and_prime()
        ready.assert_called_once_with("robot.test")
        assert sent[0] == {f"input_double_register_{i}": 0.2 for i in range(6)}
        assert sent[1]["input_int_register_0"] == 0
        assert sent[1]["input_int_register_2"] == controller._nonce
        controller.stop()
        assert sent[-1]["input_int_register_0"] == 3
        conn.disconnect.assert_called_once()


def test_running_or_bad_telemetry_never_primes_registers():
    for bad in (packet(runtime=2), packet()):
        if bad.runtime_state == 1:
            bad.tcp_offset[0] = 0.1
        controller, conn = JointServoJController(servo_config()), fake_connection()
        conn.receive.return_value = bad
        with (
            patch("ur5e_real.control.joint_servoj.require_external_motion_ready"),
            patch("ur5e_real.control.joint_servoj._imports") as imports,
        ):
            imports.return_value = fake_imports(conn)
            with pytest.raises(RuntimeError):
                controller.connect_and_prime()
            conn.send.assert_not_called()
            conn.disconnect.assert_called_once()


def test_fresh_program_identity_requires_all_fields():
    original = packet(runtime=2)
    assert validate_program_ack(original, 7)
    assert not validate_program_ack(original, 8)
    for key in ("runtime_state", "output_int_register_0", "output_int_register_1"):
        bad = SimpleNamespace(**vars(original) | {key: 0})
        assert not validate_program_ack(bad, 7)


def test_stale_state_bad_offset_and_reversed_clock_fail_closed():
    controller = JointServoJController(servo_config())
    controller._accept_state(packet())
    with pytest.raises(RuntimeError, match="time"):
        controller._accept_state(packet())
    controller._state_at = 0
    with pytest.raises(TimeoutError, match="stale"):
        controller.get_latest_joints()


def test_joint_script_and_configuration_are_separate_from_tcp():
    config = servo_config()
    source = config.program_script.read_text()
    assert "get_inverse_kin(" not in source and "servoj(joints," in source
    assert 'rtde_set_watchdog("input_int_register_1", 10, "stop")' in source
    assert "output_int_register_2" in config.config_xml.read_text()
    assert "get_inverse_kin(" in (REPOSITORY / "robot_programs/servoj_control_loop.script").read_text()
    with pytest.raises(ValueError):
        replace(config, state_timeout_s=float("nan"))


def test_background_controller_fresh_ack_and_producer_timeout():
    config = replace(servo_config(), command_timeout_s=0.05)
    controller, conn = JointServoJController(config), fake_connection()
    counter = 0

    def receive():
        nonlocal counter
        counter += 1
        time.sleep(0.002)
        value = packet(timestamp=counter * 0.002, runtime=1 if counter == 1 else 2)
        value.output_int_register_2 = controller._nonce
        return value

    conn.receive.side_effect = receive
    with (
        patch("ur5e_real.control.joint_servoj.require_external_motion_ready"),
        patch("ur5e_real.control.joint_servoj.send_urscript") as send,
        patch("ur5e_real.control.joint_servoj._imports") as imports,
    ):
        imports.return_value = fake_imports(conn)
        controller.start()
        send.assert_called_once()
        assert controller._ready.is_set()
        controller.set_target_joints([0.21] * 6)
        assert controller._stop_event.wait(0.3)
        with pytest.raises(RuntimeError, match="producer timed out"):
            controller.get_latest_joints()
        controller.stop()


def test_stale_prime_never_sends_motion_program():
    controller = JointServoJController(servo_config())
    controller._primed = True
    controller._accept_state(packet())
    controller._state_at = 0
    with (
        patch("ur5e_real.control.joint_servoj.require_external_motion_ready"),
        patch("ur5e_real.control.joint_servoj.send_urscript") as send,
    ):
        with pytest.raises(TimeoutError, match="stale"):
            controller.start()
        send.assert_not_called()
