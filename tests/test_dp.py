import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import zarr

from ur5e_real.adapters.robotwin_dp.infer_dp import (
    InferenceConfig as DPInferenceConfig,
    RealObservationEncoder,
    observation_from_vector,
)
from ur5e_real.adapters.robotwin_dp.train_dp import TrainConfig, build_train_command, run_training
from ur5e_real.control.chunk import (
    ChunkStreamConfig,
    continuous_chunk_targets,
    interpolated_tcp_targets,
    limit_tcp_target,
)
from ur5e_real.control.gripper_policy import GripperCommandConfig, GripperPolicy
from ur5e_real.control.servoj import ServoJController, ServoJStreamConfig, render_servoj_program
from ur5e_real.control.socket_speedl import (
    SocketSpeedLConfig,
    smoothed_speedl_target,
    stretched_speedl_velocity,
)
from ur5e_real.operator import resolve_dp_checkpoint
from ur5e_real.replay import ReplayConfig


class FakeGripper:
    def __init__(self):
        self.commands = []

    def open(self):
        self.commands.append("open")

    def close(self):
        self.commands.append("close")


class DiffusionPolicyAdapterTest(unittest.TestCase):
    def test_checkpoint_short_name_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            checkpoint = data_root / "checkpoints" / "dp" / "run" / "checkpoints" / "task" / "600.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertEqual(resolve_dp_checkpoint("600", data_root), checkpoint.resolve())

    def test_checkpoint_timestamp_qualified_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            old = data_root / "checkpoints" / "dp" / "task-20260904_120000" / "600.ckpt"
            new = data_root / "checkpoints" / "dp" / "task-20260905_120000" / "600.ckpt"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            old.touch()
            new.touch()
            self.assertEqual(resolve_dp_checkpoint("20260905_120000:600", data_root), new.resolve())

    def test_observation_layout_and_rotation_continuity(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        vector = np.arange(14, dtype=np.float32)
        observation = observation_from_vector(vector, image, image)
        self.assertEqual(observation["head_cam"].shape, (3, 240, 320))
        np.testing.assert_array_equal(observation["agent_pos"], vector)

        encoder = RealObservationEncoder()
        encoder.encode([0, 0, 0, 0, 0, 3.13], 0, image, image)
        second = encoder.encode([0, 0, 0, 0, 0, -3.13], 1, image, image)
        self.assertLess(abs(float(second["agent_pos"][5]) - 3.13), 0.1)
        np.testing.assert_allclose(second["agent_pos"][:6], second["agent_pos"][7:13])
        self.assertEqual(float(second["agent_pos"][13]), 1.0)

    def test_chunk_velocity_limit_and_interpolation(self):
        config = ChunkStreamConfig(max_linear_velocity=1.0, max_angular_velocity=2.0)
        target = limit_tcp_target(np.zeros(6), [1, 0, 0, 1, 0, 0], 0.1, config)
        np.testing.assert_allclose(target, [0.1, 0, 0, 0.2, 0, 0], atol=1e-6)
        points = interpolated_tcp_targets(np.zeros(6), target, 10)
        np.testing.assert_allclose(points[-1], target, atol=1e-6)

    def test_continuous_chunk_does_not_stop_at_action_knots(self):
        config = ChunkStreamConfig(policy_hz=10.0, servo_hz=500, max_linear_velocity=0.05)
        samples, waypoints, steps = continuous_chunk_targets(
            np.zeros(6),
            [[0.005, 0, 0, 0, 0, 0], [0.010, 0, 0, 0, 0, 0]],
            config,
        )
        path = np.vstack((np.zeros(6), samples))
        velocity = np.diff(path[:, :3], axis=0) * config.servo_hz
        self.assertEqual(steps, 50)
        self.assertEqual(len(samples), 100)
        np.testing.assert_allclose(waypoints[-1], [0.010, 0, 0, 0, 0, 0], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(velocity[49])), 0.05, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(velocity[50])), 0.05, places=5)
        self.assertLessEqual(float(np.linalg.norm(velocity, axis=1).max()), 0.05001)

    def test_socket_speedl_smoothing_is_selectable_and_bounded(self):
        current = np.zeros(6, dtype=np.float32)
        desired = np.asarray([0.010, 0, 0, 0, 0, 0], dtype=np.float32)
        smooth_target, smooth_velocity = smoothed_speedl_target(
            current,
            desired,
            None,
            SocketSpeedLConfig(smoothing_alpha=0.5, max_linear_velocity=0.05),
        )
        raw_target, raw_velocity = smoothed_speedl_target(
            current,
            desired,
            None,
            SocketSpeedLConfig(smoothing_alpha=1.0, max_linear_velocity=0.05),
        )
        default_target, _ = smoothed_speedl_target(
            current, desired, None, SocketSpeedLConfig(smoothing_alpha=1.0)
        )
        self.assertAlmostEqual(float(smooth_target[0]), 0.0025, places=6)
        self.assertAlmostEqual(float(raw_target[0]), 0.005, places=6)
        self.assertAlmostEqual(float(default_target[0]), 0.010, places=6)
        self.assertLessEqual(float(np.linalg.norm(smooth_velocity[:3])), 0.05001)
        self.assertLessEqual(float(np.linalg.norm(raw_velocity[:3])), 0.05001)

    def test_gripper_hysteresis(self):
        gripper = FakeGripper()
        policy = GripperPolicy(
            gripper,
            GripperCommandConfig(stable_count=2, minimum_command_interval_s=0.0),
        )
        policy.step(0.7, now=1.0)
        policy.step(0.7, now=1.1)
        policy.step(0.3, now=1.2)
        policy.step(0.3, now=1.3)
        self.assertEqual(gripper.commands, ["close", "open"])

    def test_socket_stretch_spreads_only_the_final_delta_over_the_gap(self):
        velocity, duration_s = stretched_speedl_velocity(
            np.zeros(6),
            [0.02, 0, 0, 0, 0, 0],
            0.08,
            SocketSpeedLConfig(policy_hz=10.0),
        )
        self.assertAlmostEqual(duration_s, 0.20, places=6)
        self.assertAlmostEqual(float(velocity[0]), 0.10, places=6)

    def test_servoj_tuning_is_rendered_without_changing_the_template(self):
        source = Path("robot_programs/servoj_control_loop.script").read_text(encoding="utf-8")
        configured = render_servoj_program(source, 0.2, 400)
        self.assertIn("local servoj_lookahead_time = 0.200000", configured)
        self.assertIn("local servoj_gain = 400", configured)
        self.assertIn("local servoj_lookahead_time = 0.1", source)
        self.assertIn("local servoj_gain = 300", source)

    def test_servoj_primes_persistent_watchdog_before_activation(self):
        watchdog_modes = []

        class FakeConnection:
            def __init__(self, _host, _port):
                self.receive_count = 0

            def connect(self):
                pass

            def get_controller_version(self):
                pass

            def send_output_setup(self, _names, _types, _frequency):
                pass

            def send_input_setup(self, names, _types):
                if names == ["watchdog"]:
                    return SimpleNamespace(input_int_register_0=3)
                return SimpleNamespace(**{f"input_double_register_{index}": 0.0 for index in range(6)})

            def send_start(self):
                return True

            def receive(self):
                self.receive_count += 1
                runtime_state = 1 if self.receive_count == 1 else 2
                return SimpleNamespace(actual_TCP_pose=[0.0] * 6, runtime_state=runtime_state)

            def send(self, recipe):
                if hasattr(recipe, "input_int_register_0"):
                    watchdog_modes.append(recipe.input_int_register_0)

            def send_pause(self):
                pass

            def disconnect(self):
                pass

        class FakeConfigFile:
            def __init__(self, _path):
                pass

            def get_recipe(self, name):
                return ([name], ["DOUBLE"])

        with tempfile.TemporaryDirectory() as directory:
            config_xml = Path(directory) / "rtde.xml"
            config_xml.touch()
            controller = ServoJController(ServoJStreamConfig("robot.local", config_xml))
            with mock.patch(
                "ur5e_real.control.servoj._imports",
                return_value=(
                    SimpleNamespace(RTDE=FakeConnection),
                    SimpleNamespace(ConfigFile=FakeConfigFile),
                ),
            ):
                controller.connect_and_prime()
            self.assertEqual(watchdog_modes, [0])
            self.assertFalse(controller._running)
            controller.activate()
            self.assertEqual(watchdog_modes, [0, 2])
            self.assertTrue(controller._running)
            controller.stop()
            self.assertEqual(watchdog_modes, [0, 2, 3])

    def test_dp_inference_defaults_to_proven_rtde_configuration(self):
        config = DPInferenceConfig()
        self.assertEqual(config.backend, "rtde")
        self.assertEqual(config.chunks, 0)
        self.assertEqual(config.max_linear_velocity, 0.40)
        self.assertEqual(config.diffusion_steps, 10)
        self.assertEqual(config.servoj_lookahead_time, 0.1)
        self.assertEqual(config.servoj_gain, 300)

    def test_rtde_replay_defaults_match_dp_action_executor(self):
        inference = DPInferenceConfig()
        replay = ReplayConfig(Path("actions.csv"), None, "robot.local")
        self.assertEqual(replay.chunk_size, inference.n_action_steps)
        self.assertEqual(replay.policy_hz, inference.policy_hz)
        self.assertEqual(replay.rtde_max_linear_velocity, inference.max_linear_velocity)
        self.assertEqual(replay.servoj_lookahead_time, inference.servoj_lookahead_time)
        self.assertEqual(replay.servoj_gain, inference.servoj_gain)

    def test_training_command_preserves_upstream_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robotwin = root / "RoboTwin"
            train_script = robotwin / "policy" / "DP" / "train.py"
            train_script.parent.mkdir(parents=True)
            train_script.write_text("", encoding="utf-8")
            dataset_path = root / "episode.zarr"
            dataset = zarr.open_group(str(dataset_path), mode="w")
            data = dataset.create_group("data")
            data.create_dataset("state", data=np.zeros((5, 14), dtype=np.float32))
            command = build_train_command(
                robotwin,
                dataset_path,
                root / "output",
                TrainConfig("task", "simple", 1, debug=True),
            )
            self.assertIn("--config-name=robot_dp_14.yaml", command)
            self.assertIn("task.dataset.val_ratio=0.0", command)
            self.assertIn("dataloader.batch_size=5", command)
            self.assertIn("training.debug=true", command)

    def test_training_batch_fits_shortest_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robotwin = root / "RoboTwin"
            train_script = robotwin / "policy" / "DP" / "train.py"
            train_script.parent.mkdir(parents=True)
            train_script.write_text("", encoding="utf-8")
            dataset_path = root / "episodes.zarr"
            dataset = zarr.open_group(str(dataset_path), mode="w")
            data = dataset.create_group("data")
            data.create_dataset("state", data=np.zeros((12, 14), dtype=np.float32))
            meta = dataset.create_group("meta")
            meta.create_dataset("episode_ends", data=np.asarray([7, 12], dtype=np.int64))
            command = build_train_command(
                robotwin,
                dataset_path,
                root / "output",
                TrainConfig("task", "simple", 2),
            )
            self.assertIn("dataloader.batch_size=5", command)
            self.assertIn("val_dataloader.batch_size=5", command)

    @mock.patch("ur5e_real.adapters.robotwin_dp.train_dp.subprocess.run")
    def test_training_runs_from_output_directory(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robotwin = root / "RoboTwin"
            train_script = robotwin / "policy" / "DP" / "train.py"
            train_script.parent.mkdir(parents=True)
            train_script.write_text("", encoding="utf-8")
            dataset_path = root / "episode.zarr"
            dataset = zarr.open_group(str(dataset_path), mode="w")
            data = dataset.create_group("data")
            data.create_dataset("state", data=np.zeros((5, 14), dtype=np.float32))
            output = root / "output"
            run_training(
                robotwin,
                dataset_path,
                TrainConfig("task", "simple", 1, debug=True),
                output_dir=output,
            )
            self.assertEqual(run.call_args.kwargs["cwd"], output)
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
