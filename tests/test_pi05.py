import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
