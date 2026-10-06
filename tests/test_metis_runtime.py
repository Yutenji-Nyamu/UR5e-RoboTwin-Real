"""Metis data/runtime boundaries. No physical camera, serial or motion commands."""

import json
import itertools
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ur5e_real.adapters.metiswam4d import runtime
from ur5e_real.adapters.metiswam4d.contract import fit_stats, require_same_contract
from ur5e_real.adapters.metiswam4d.data import image_paths, prepare, read_dataset
from ur5e_real.adapters.metiswam4d.server import validate_observation
from ur5e_real.control.gripper_policy import GripperCommandConfig, GripperPolicy
from test_metiswam4d import episode


@pytest.fixture
def dataset(tmp_path):
    run, _ = episode(tmp_path)
    output = tmp_path / "prepared"
    prepare(tmp_path, [run], "drawer", output)
    return output


def arguments(dataset, **overrides):
    return SimpleNamespace(**(dict(dataset=dataset, data_root=None, index=0, mode="offline", execute=False,
                                  lab_config=None, port=8006, timeout=1.5, action_steps=6, chunks=1, speed=0.6,
                                  expected_instance_id=None, shadow_gripper=None, output=None) | overrides))


def test_data_relocation_tamper_check_and_train_only_stats(tmp_path):
    run, _ = episode(tmp_path)
    held, _ = episode(tmp_path, "20261005_130000")
    csv = tmp_path / "raw/action" / f"rtde_tcp_gripper_{held}.csv"
    csv.write_text(csv.read_text().replace("0.02", "0.09"))
    out = tmp_path / "dataset"
    result = prepare(tmp_path, [run, held], "drawer", out, validation_runs=[held])
    assert result["training_windows"] == result["validation_windows"] == 3
    contract, states, actions, valid, images = read_dataset(out)
    assert contract["stats"] == fit_stats(states[:3], actions[:3], valid[:3])
    newroot = tmp_path / "restored"
    newroot.mkdir()
    shutil.move(tmp_path / "raw", newroot / "raw")
    assert not Path(images[0]["head"]).is_absolute()
    assert all(p.is_file() for p in image_paths(out, images[0], data_root=newroot))
    require_same_contract(contract, read_dataset(out)[0])
    with (out / "images.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="asset differs"):
        read_dataset(out)


def test_reject_inconsistent_initial_postures(tmp_path):
    run, _ = episode(tmp_path)
    other, path = episode(tmp_path, "20261005_130000")
    meta = json.loads(path.read_text())
    meta["initial_robot_state"]["actual_q"][0] = 0.1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="disagree on home"):
        prepare(tmp_path, [run, other], "drawer", tmp_path / "prepared")


def test_camera_color_and_joint7_contract():
    bgr = np.array([[[10, 20, 30]]], np.uint8)
    obs = runtime.observation(np.arange(6), 1, bgr, bgr)
    np.testing.assert_array_equal(obs["state"], [0, 1, 2, 3, 4, 5, 1])
    np.testing.assert_array_equal(obs["head"], [[[30, 20, 10]]])
    validate_observation(obs)
    bgr[:] = 0
    assert obs["head"].any()
    with pytest.raises(ValueError):
        validate_observation(dict(obs, state=np.zeros(14)))


def test_gripper_can_finish_closed_and_start_closed():
    grip = MagicMock()
    cfg = GripperCommandConfig(stable_count=2, minimum_command_interval_s=0.5, maximum_cycles=None)
    policy = GripperPolicy(grip, cfg)
    for i, g in enumerate([1, 1, 0, 0, 1, 1]):
        policy.step(g, now=10 + i)
    assert [c[0] for c in grip.method_calls] == ["close", "open", "close"]
    assert policy.estimated == 1 and policy.cycles == 1
    policy = GripperPolicy(grip, cfg, initial_state=1)
    policy.step(0, now=20)
    policy.step(0, now=21)
    assert policy.estimated == 0


def test_default_offline_and_home_do_not_open_hardware(dataset, monkeypatch):
    for name in ("FreshCameras", "RtdeStateClient", "JointServoJController", "GripperSerial"):
        monkeypatch.setattr(runtime, name, MagicMock(side_effect=AssertionError("hardware opened")))
    client = MagicMock()
    client.__enter__.return_value = client
    client.infer.return_value = {"actions": np.zeros((50, 7), np.float32), "client_elapsed_s": 0.01}
    monkeypatch.setattr(runtime, "PolicyClient", lambda *a, **k: client)
    assert runtime.run(arguments(dataset))["physical_execution"] is False
    assert runtime.home(arguments(dataset))["physical_execution"] is False


def mock_live(monkeypatch, *, status="SFT_not_physical_validation", fail_warmup=False):
    client = MagicMock()
    client.__enter__.return_value = client
    client.metadata = {"training_status": status}
    actions = np.zeros((50, 7), np.float32)
    actions[:6, 6] = [1, 1, 0, 0, 1, 1]
    client.infer.return_value = {"actions": actions, "client_elapsed_s": 0.01}
    if fail_warmup:
        client.infer.side_effect = TimeoutError("warmup failed")
    monkeypatch.setattr(runtime, "PolicyClient", lambda *a, **k: client)
    lab = SimpleNamespace(robot=SimpleNamespace(host="127.0.0.1", rtde_port=30004), cameras=None,
                          gripper=SimpleNamespace(port="mock", baudrate=9600, timeout_s=1))
    monkeypatch.setattr(runtime, "load_config", lambda _: lab)
    cameras = MagicMock()
    cameras.__enter__.return_value = cameras
    cameras.read.return_value = SimpleNamespace(head=np.zeros((8, 12, 3), np.uint8), wrist=np.zeros((8, 12, 3), np.uint8))
    monkeypatch.setattr(runtime, "FreshCameras", lambda _: cameras)
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.receive_state.return_value = SimpleNamespace(actual_q=[0] * 6)
    monkeypatch.setattr(runtime, "RtdeStateClient", lambda _: reader)
    controller = MagicMock()
    controller.get_latest_state.return_value = reader.receive_state.return_value
    controller.get_latest_joints.return_value = controller.get_commanded_joints.return_value = [0] * 6
    factory = MagicMock(return_value=controller)
    monkeypatch.setattr(runtime, "JointServoJController", factory)
    monkeypatch.setattr(runtime, "controller_config", lambda *a, **k: None)
    gripper = MagicMock()
    gripper.__enter__.return_value = gripper
    grip_factory = MagicMock(return_value=gripper)
    monkeypatch.setattr(runtime, "GripperSerial", grip_factory)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    return client, factory, grip_factory


def test_shadow_and_warmup_failure_never_create_actuators(dataset, monkeypatch):
    _, control, grip = mock_live(monkeypatch)
    runtime.run(arguments(dataset, mode="shadow", lab_config=Path("unused")))
    control.assert_not_called()
    grip.assert_not_called()
    client, control, grip = mock_live(monkeypatch, fail_warmup=True)
    with pytest.raises(TimeoutError):
        runtime.run(arguments(dataset, mode="execute", lab_config=Path("unused")))
    control.assert_not_called()
    grip.assert_not_called()


def test_execute_all_chunks_including_after_release_and_stop_on_failure(dataset, monkeypatch):
    client, control, grip = mock_live(monkeypatch)
    ticks = itertools.count(10)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    def stream(controller, targets, motion, *, on_waypoint):
        assert len(targets) == 6
        for i in range(6):
            on_waypoint(i, targets[i])
        return {"waypoints": 6}
    monkeypatch.setattr(runtime, "stream_joint_chunk", stream)
    runtime.run(arguments(dataset, mode="execute", lab_config=Path("unused"), chunks=2))
    commands = [c[0] for c in grip.return_value.method_calls if c[0] in ("open", "close")]
    assert commands == ["open", "close", "open", "close", "open", "close"]
    assert client.infer.call_count == 3  # warmup + both chunks; no stop at the intermediate release
    control.return_value.stop.assert_called_once()
    client, control, _ = mock_live(monkeypatch)
    client.infer.side_effect = [client.infer.return_value, TimeoutError("late")]
    with pytest.raises(TimeoutError):
        runtime.run(arguments(dataset, mode="execute", lab_config=Path("unused")))
    control.return_value.stop.assert_called_once()


def test_cli_robot_side_does_not_import_model_frameworks():
    script = "from ur5e_real.adapters.metiswam4d import __main__, runtime; import sys; " \
             "assert 'torch' not in sys.modules and 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_vae_relocation_requires_identical_assets(tmp_path):
    pytest.importorskip("torch")
    from ur5e_real.adapters.metiswam4d.checkpoint import vae_identity, verify_vae

    vae = tmp_path / "vae"
    vae.mkdir()
    (vae / "config.json").write_text("{}")
    (vae / "weights.safetensors").write_bytes(b"fake")
    identity = vae_identity(vae)
    moved = tmp_path / "moved"
    vae.rename(moved)
    saved = {"vae_path": str(vae), "vae_sha256": identity}
    assert verify_vae(saved, moved) == moved
    (moved / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="VAE assets differ"):
        verify_vae(saved, moved)
