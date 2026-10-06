"""Single-arm semantics and full-task windows, independent of private upstream code."""

import csv
import json

import numpy as np
import pytest

from ur5e_real.adapters.metiswam4d.contract import (
    SLOTS, absolute_actions, delta_actions, fit_stats, normalize, to_pi05, to_unified,
)
from ur5e_real.adapters.metiswam4d.data import prepare


def test_joint_delta_and_normalization_round_trip():
    rng = np.random.default_rng(3)
    state = rng.normal(size=(2, 7)).astype(np.float32)
    action = rng.normal(size=(2, 50, 7)).astype(np.float32)
    action[..., 6] = 1
    valid = np.ones((2, 50), bool)
    valid[1, 10:] = False
    delta = delta_actions(state, action)
    np.testing.assert_array_equal(delta[..., 6], action[..., 6])
    stats = fit_stats(state, action, valid)
    recovered = normalize(normalize(delta, stats, "action"), stats, "action", inverse=True)
    np.testing.assert_allclose(absolute_actions(state, recovered), action, atol=5e-7)
    invalid_changed = action.copy()
    invalid_changed[~valid] = 5000
    assert stats == fit_stats(state, invalid_changed, valid)


def test_mapping_keeps_one_physical_arm_and_masks_unused_slots():
    physical = np.arange(21, dtype=np.float32).reshape(3, 7)
    physical[:, 6] = [0, 1, 0]
    native = to_unified(physical)
    np.testing.assert_array_equal(native[:, list(SLOTS)], physical)
    assert not native[:, [i for i in range(80) if i not in SLOTS]].any()
    compat = to_pi05(physical)
    np.testing.assert_array_equal(compat[:, :6], physical[:, :6])
    np.testing.assert_array_equal(compat[:, 7:13], physical[:, :6])
    np.testing.assert_array_equal(compat[:, 13], physical[:, 6])
    assert not compat[:, 6].any()


@pytest.mark.parametrize("values", [np.zeros(14), [np.nan] * 7, [np.inf] * 7])
def test_reject_bad_physical_vectors(values):
    with pytest.raises(ValueError):
        to_unified(values)


def episode(root, run="20261005_120000"):
    import cv2

    action = root / "raw" / "action"
    action.mkdir(parents=True, exist_ok=True)
    camera = root / "raw" / "camera" / f"cam_dual_{run}"
    for role in ("head", "wrist"):
        (camera / role).mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(str(camera / role / f"frame_{i+1:05d}.png"), np.full((8, 12, 3), i, np.uint8))
    meta = {"run_id": run, "schema_version": 3, "task": "drawer", "outcome": "success",
            "recording_status": "completed", "state_recording": {"joint_source": "actual_q",
            "joint_position_unit": "rad", "joint_order": ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]},
            "initial_robot_state": {"actual_q": [0] * 6, "tcp_offset": [0] * 6, "tcp_pose": [0] * 6},
            "collection": {"initial_gripper_state": "open"}}
    manifest = action / f"session_{run}.json"
    manifest.write_text(json.dumps(meta))
    with (action / f"rtde_tcp_gripper_{run}.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["controller_time_s", *[f"actual_q_{i}" for i in range(6)], "gripper_state",
                         "tcp_x", "tcp_y", "tcp_z"])
        writer.writerows([[i / 10, *[i / 100] * 6, grip, 0, 0, 0] for i, grip in enumerate([0, 1, 0, 1])])
    with (action / f"sync_action_cam_{run}.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["controller_time_s", "frame_idx", "head_image", "wrist_image"])
        writer.writerows([[i / 10, i+1, *[f"frame_{i+1:05d}.png"] * 2] for i in range(4)])
    (action / f"gripper_events_{run}.csv").write_text(
        "controller_time_s,event,gripper_state\n0.1,close,1\n0.2,open,0\n0.3,close,1\n")
    return run, manifest


def test_full_episode_preserves_final_close_and_masks_padding(tmp_path):
    run, _ = episode(tmp_path)
    out = tmp_path / "prepared"
    report = prepare(tmp_path, [run], "drawer", out)
    assert report["training_windows"] == 3
    with np.load(out / "actions.npz", allow_pickle=False) as values:
        np.testing.assert_array_equal(values["actions"][0, :3, 6], [1, 0, 1])
        np.testing.assert_array_equal(values["valid"].sum(axis=1), [3, 2, 1])
        assert values["actions"][0, 2, 0] == pytest.approx(0.03)
    with pytest.raises(FileExistsError):
        prepare(tmp_path, [run], "drawer", out)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("outcome", "failure"), ("recording_status", "failed")])
def test_reject_unusable_episode(tmp_path, field, value):
    run, path = episode(tmp_path)
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="completed success"):
        prepare(tmp_path, [run], "drawer", tmp_path / "prepared")


def test_native_mask_loss_and_checkpoint_guard(tmp_path):
    torch = pytest.importorskip("torch")
    from pathlib import Path
    from ur5e_real.adapters.metiswam4d.checkpoint import load, save
    from ur5e_real.adapters.metiswam4d.contract import VERSION
    from ur5e_real.adapters.metiswam4d.native import build_model, load_source, model_config, training_precision
    from ur5e_real.adapters.metiswam4d.policy import ActionPolicy

    source_path = Path(__file__).resolve().parents[1] / ".third_party/MetisWAM4D"
    if not source_path.exists():
        pytest.skip("private source snapshot is optional")
    source = load_source(source_path)
    torch.set_num_threads(2)
    config = model_config(tiny=True)
    model = training_precision(build_model(config), device="cpu", dtype=torch.bfloat16)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    assert parameter.dtype == torch.float32
    assert next(model.video.parameters()).dtype == torch.bfloat16
    with torch.no_grad():
        parameter.view(-1)[0] = 0.1
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert parameter.view(-1)[0].item() < 0.1 - 5e-6
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    state, action = np.zeros((1, 7), np.float32), np.zeros((1, 50, 7), np.float32)
    valid = np.zeros((1, 50), bool)
    valid[:, :2] = True
    stats = fit_stats(state, action, valid)
    policy = ActionPolicy(model, stats)
    latent = torch.rand(1, 4, 1, 8, 8)
    torch.manual_seed(6)
    loss = policy.loss(latent, state, action, valid)
    action[:, 2:] = 100
    torch.manual_seed(6)
    torch.testing.assert_close(loss, policy.loss(latent, state, action, valid))
    contract = {"version": VERSION, "fps": 10, "horizon": 50, "slots": list(SLOTS), "stats": stats}
    path = tmp_path / "tiny.pt"
    save(path, model, config, contract, source)
    restored, metadata = load(path, source, allow_tiny=True)
    assert metadata["trainable_dtype"] == "float32"
    for before, after in zip(model.parameters(), restored.model.parameters()):
        assert before.dtype == after.dtype
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    with pytest.raises(ValueError, match="diagnostic"):
        load(path, source)
    with pytest.raises(ValueError, match="source differs"):
        load(path, "wrong", allow_tiny=True)
