import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ur5e_real.adapters.robotwin_pi05 import infer as pi05_infer
from ur5e_real.adapters.robotwin_pi05.config import action_expert_path
from ur5e_real.adapters.robotwin_pi05.contract import (
    JOINT_ORDER,
    NATIVE_PATCH,
    ROBOTWIN_COMMIT,
    STATE_LAYOUT,
    decode_actions,
    encode_state,
    require_home,
    require_same_contract,
    validate_contract,
)
from ur5e_real.adapters.robotwin_pi05.dataset import future_actions
from ur5e_real.adapters.robotwin_pi05.native import REPOSITORY
from ur5e_real.adapters.robotwin_pi05.process_data import crop_bounds, load_selection, read_episode
from ur5e_real.adapters.robotwin_pi05.runtime import FreshCameras, observation, prepare


def contract():
    return {
        "version": 1,
        "policy": "pi05",
        "action_space": "joint_position",
        "state_layout": STATE_LAYOUT,
        "joint_unit": "rad",
        "joint_order": list(JOINT_ORDER),
        "fps": 10,
        "action_horizon": 50,
        "adapt_to_pi": False,
        "delta_joint_actions": True,
        "robotwin_commit": ROBOTWIN_COMMIT,
        "native_patch": NATIVE_PATCH,
        "dataset_id": "org/test",
        "home_q": [0.0] * 6,
        "joint_lower": [-4.0] * 6,
        "joint_upper": [4.0] * 6,
        "tcp_lower": [-1.0] * 3,
        "tcp_upper": [1.0] * 3,
        "tcp_offset": [0.0] * 6,
    }


def test_joint_contract_cannot_accept_tcp_or_wrong_units():
    validate_contract(contract())
    for key, value in (
        ("action_space", "tcp_pose"),
        ("joint_unit", "deg"),
        ("adapt_to_pi", True),
        ("delta_joint_actions", False),
        ("joint_order", list(reversed(JOINT_ORDER))),
    ):
        with pytest.raises(ValueError):
            validate_contract(contract() | {key: value})
    with pytest.raises(ValueError):
        require_same_contract(contract(), contract() | {"dataset_id": "org/other"})
    with pytest.raises(RuntimeError):
        require_home([0.1] * 6, contract())


def test_14d_layout_next_state_labels_and_absolute_decoding():
    q = np.array([3.3, -3.4, 1, 2, 0.1, -0.2])
    vector = encode_state(q, 1)
    np.testing.assert_allclose(vector[:6], q)
    np.testing.assert_allclose(vector[7:13], q)
    assert vector[6] == 0 and vector[13] == 1
    vectors = np.stack([encode_state(q + i * 0.01, i > 1) for i in range(4)])
    future = future_actions(vectors)
    np.testing.assert_equal(future[0, 0], vectors[1])
    np.testing.assert_equal(future[0, 2], vectors[3])
    np.testing.assert_equal(future[-1, -1], vectors[3])
    predicted = np.tile(vector, (50, 1))
    predicted[:, 7:13] = 99  # The physical single arm comes only from the first six outputs.
    joints, grip = decode_actions(predicted)
    np.testing.assert_allclose(joints[0], q)
    assert np.all(grip == 1)
    with pytest.raises(ValueError):
        decode_actions(np.zeros((50, 32)))


def test_only_native_action_expert_and_projections_are_trainable():
    for path in (
        ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
        ("PaliGemma", "llm", "final_norm_1", "scale"),
        ("action_in_proj", "kernel"),
        ("time_mlp_out", "bias"),
    ):
        assert action_expert_path(path)
    for path in (
        ("PaliGemma", "img", "Transformer", "encoderblock_1", "kernel"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
        ("PaliGemma", "llm", "embedder", "input_embedding"),
    ):
        assert not action_expert_path(path)


def test_live_rgb_conversion_and_missing_camera_not_duplicated():
    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    obs = observation(np.zeros(6), 0, bgr, bgr, "cube")
    assert set(obs["images"]) == {"cam_high", "cam_right_wrist"}
    assert obs["images"]["cam_high"].shape == (3, 4, 5)
    assert np.all(obs["images"]["cam_high"][0] == 255)


def test_crop_respects_gripper_when_joints_are_static():
    q = np.zeros((20, 6))
    grip = np.r_[np.zeros(5), np.ones(10), np.zeros(5)]
    assert crop_bounds(q, grip, 0.005, 3, 3) == (2, 18)


def test_v2_or_unreviewed_episode_rejected_before_reading_any_actions(tmp_path):
    selection = load_selection(REPOSITORY / "configs/pi05_cube_joint_5.json")
    action = tmp_path / "raw/action"
    action.mkdir(parents=True)
    run = selection["run_ids"][0]
    # Synthetic fixtures are test data, not edits to the user's recordings.
    (action / f"session_{run}.json").write_text(json.dumps({"run_id": run, "schema_version": 2}))
    with pytest.raises(ValueError, match="v3"):
        read_episode(tmp_path, run, selection)


def test_prepare_dry_run_never_constructs_a_controller():
    with patch("ur5e_real.adapters.robotwin_pi05.runtime.JointServoJController") as controller:
        prepare(MagicMock(), contract(), execute=False)
        controller.assert_not_called()


def test_camera_cache_rejects_stale_delivery():
    cameras = FreshCameras(MagicMock())
    cameras._pair = object()
    cameras._at = 0
    with pytest.raises(TimeoutError):
        cameras.read()


def test_all_new_cli_modules_import_without_model_libraries():
    import importlib

    for name in ("infer", "serve", "train", "process_data", "__main__"):
        importlib.import_module(f"ur5e_real.adapters.robotwin_pi05.{name}")


@pytest.mark.parametrize("extra, expected", [([], 20), (["--action-steps", "6"], 6), (["--action-steps", "50"], 50)])
def test_inference_cli_defaults_to_user_selected_prefix_without_loading_a_model(extra, expected):
    with patch("sys.argv", ["pi05-infer", "--dataset", "unused", *extra]), patch.object(pi05_infer, "run") as run:
        pi05_infer.main()
    assert run.call_args.args[0].action_steps == expected
    assert run.call_args.args[0].mode == "offline"


def test_shadow_selects_twenty_of_fifty_predictions_and_records_the_actual_settings(tmp_path):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    actions = np.stack([encode_state(np.full(6, i * 0.01), 0) for i in range(1, 51)])
    output = tmp_path / "shadow.json"
    argv = [
        "pi05-infer",
        "--dataset",
        "unused",
        "--lab-config",
        "unused",
        "--mode",
        "shadow",
        "--chunks",
        "1",
        "--output",
        str(output),
    ]
    with (
        patch("sys.argv", argv),
        patch.object(pi05_infer, "validate_dataset", return_value=(contract() | {"prompt": "cube"}, {})),
        patch.object(pi05_infer, "load_config", return_value=MagicMock()),
        patch.object(pi05_infer, "PolicyClient") as client,
        patch.object(pi05_infer, "FreshCameras") as cameras,
        patch.object(pi05_infer, "RtdeStateClient") as reader,
        patch.object(pi05_infer, "JointServoJController") as controller,
        patch.object(pi05_infer, "GripperSerial") as gripper,
        patch.object(pi05_infer, "plan_joint_chunk", wraps=pi05_infer.plan_joint_chunk) as plan,
    ):
        client.return_value.__enter__.return_value.infer.return_value = {"actions": actions, "client_elapsed_s": 0.1}
        cameras.return_value.__enter__.return_value.read.return_value = SimpleNamespace(head=image, wrist=image)
        reader.return_value.__enter__.return_value.receive_state.return_value = SimpleNamespace(actual_q=np.zeros(6))
        pi05_infer.main()
        np.testing.assert_allclose(plan.call_args.args[1], actions[:20, :6])
        controller.assert_not_called()
        gripper.assert_not_called()
    report = json.loads(output.read_text())[0]
    assert report["action_horizon"] == 50
    assert report["action_steps"] == 20
    assert report["policy_hz"] == 10
    assert report["execution_preflight"] == "pass"


def test_replaced_model_service_is_rejected_before_hardware_access():
    args = SimpleNamespace(
        dataset="unused", action_steps=20, chunks=30, speed=0.6, port=18005, timeout=1.5, expected_instance_id="owned"
    )
    with (
        patch.object(pi05_infer, "validate_dataset", return_value=(contract(), {})),
        patch.object(pi05_infer, "PolicyClient") as client,
        patch.object(pi05_infer, "FreshCameras") as cameras,
    ):
        client.return_value.__enter__.return_value.metadata = {"instance_id": "different"}
        with pytest.raises(RuntimeError, match="instance changed"):
            pi05_infer.run(args)
        cameras.assert_not_called()


@pytest.mark.parametrize("cycles,k,executed", [(1, 20, [8]), (2, 20, [20, 4]), (2, 50, [24])])
def test_execute_finishes_only_after_contract_cycles_with_final_hold(tmp_path, cycles, k, executed):
    from ur5e_real.control.joint import stream_joint_chunk as real_stream

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    selected = contract() | {"prompt": "stack"}
    if cycles != 1:  # Verify legacy metadata without a count still finishes after one cycle.
        selected["gripper_cycles"] = cycles
    state = SimpleNamespace(actual_q=np.zeros(6))
    timeline = np.r_[np.ones(6), np.zeros(10), np.ones(6), np.zeros(100)]
    offsets = iter([0, *range(0, 3 * k, k)])  # First RPC is the live-shape warmup.
    clock = [10.0]

    def sleep(seconds):
        clock[0] += seconds

    def prediction(_obs):
        offset = next(offsets)
        actions = np.stack([encode_state(np.zeros(6), value) for value in timeline[offset : offset + 50]])
        return {"actions": actions, "client_elapsed_s": 0.1}

    def stream(controller, targets, motion, **options):
        return real_stream(controller, targets, motion, clock=lambda: clock[0], sleep=sleep, **options)

    output = tmp_path / "execution.json"
    args = SimpleNamespace(
        dataset="unused",
        mode="execute",
        lab_config="unused",
        action_steps=k,
        chunks=3,
        speed=0.6,
        port=18005,
        timeout=1.5,
        output=output,
    )
    with (
        patch.object(pi05_infer, "validate_dataset", return_value=(selected, {})),
        patch.object(pi05_infer, "load_config", return_value=MagicMock()),
        patch.object(pi05_infer, "PolicyClient") as client,
        patch.object(pi05_infer, "FreshCameras") as cameras,
        patch.object(pi05_infer, "RtdeStateClient") as reader,
        patch.object(pi05_infer, "JointServoJController") as controller,
        patch.object(pi05_infer, "GripperSerial") as serial,
        patch.object(pi05_infer.time, "sleep", side_effect=sleep),
        patch("ur5e_real.control.gripper_policy.time.monotonic", side_effect=lambda: clock[0]),
        patch.object(pi05_infer, "stream_joint_chunk", side_effect=stream) as streamed,
    ):
        policy = client.return_value.__enter__.return_value
        policy.metadata = {"training_status": "SFT_not_physical_validation"}
        policy.infer.side_effect = prediction
        cameras.return_value.__enter__.return_value.read.return_value = SimpleNamespace(head=image, wrist=image)
        reader.return_value.__enter__.return_value.receive_state.return_value = state
        robot = controller.return_value
        robot.action_space = "joint_position"
        robot.get_latest_state.return_value = state
        robot.get_latest_joints.return_value = np.zeros(6)
        robot.get_commanded_joints.return_value = np.zeros(6)
        pi05_infer.run(args)
        gripper = serial.return_value.__enter__.return_value
        assert gripper.close.call_count == cycles
        assert gripper.open.call_count == cycles + 1  # Includes explicit initial open.
        assert streamed.call_count == len(executed) + 1
        assert len(streamed.call_args.args[1]) == 10  # One second hold only after the final release.
        assert not streamed.call_args.kwargs
        robot.stop.assert_called_once()
    reports = json.loads(output.read_text())
    assert [row["executed_waypoints"] for row in reports] == executed
    assert reports[-1]["gripper_cycles_completed"] == cycles
    assert all(row["gripper_cycles_required"] == cycles for row in reports)
    if len(reports) > 1:
        assert reports[0]["gripper_cycles_completed"] == 1
