from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ur5e_real import pi05_operator as operator


def setup_trial(monkeypatch, tmp_path):
    trial = SimpleNamespace(
        lab=MagicMock(),
        checkpoint=tmp_path / "run" / "1000",
        dataset=tmp_path / "data",
        contract={"dataset_id": "org/cube"},
        verification={"status": "SFT_not_physical_validation"},
    )
    model_python = tmp_path / "python"
    model_python.touch()
    model_python.chmod(0o755)
    monkeypatch.setattr(operator, "REPOSITORY", tmp_path)
    monkeypatch.setattr(operator, "MODEL_PYTHON", model_python)
    monkeypatch.setattr(operator, "_enter_repository", lambda: None)
    monkeypatch.setattr(operator, "load_trial", lambda _: trial)
    return trial


def test_checkpoint_reference_resolves_unique_run_and_rejects_ambiguity(tmp_path):
    first = tmp_path / "joint5_sft_20260909_01" / "1000"
    first.mkdir(parents=True)
    assert operator.resolve_checkpoint("20260909_01:1000", tmp_path) == first
    assert operator.resolve_checkpoint(str(first), tmp_path) == first
    (tmp_path / "another_run" / "1000").mkdir(parents=True)
    with pytest.raises(ValueError, match="2 directories"):
        operator.resolve_checkpoint("1000", tmp_path)
    with pytest.raises(ValueError, match="0 directories"):
        operator.resolve_checkpoint("missing:1000", tmp_path)
    with pytest.raises(ValueError, match="checkpoint must"):
        operator.resolve_checkpoint("latest", tmp_path)


@pytest.mark.parametrize("dry_run", [True, False])
def test_init_uses_joint_prepare_and_dry_run_never_checks_hardware(monkeypatch, tmp_path, dry_run):
    trial = setup_trial(monkeypatch, tmp_path)
    prepare, doctor = MagicMock(), MagicMock(return_value=[])
    monkeypatch.setattr(operator, "prepare", prepare)
    monkeypatch.setattr(operator, "run_doctor", doctor)
    monkeypatch.setattr("sys.argv", ["ur5e-pi05-infer-init", *(["--dry-run"] if dry_run else [])])
    assert operator.infer_init() == 0
    prepare.assert_called_once_with(trial.lab, trial.contract, execute=not dry_run)
    if dry_run:
        doctor.assert_not_called()
    else:
        doctor.assert_called_once_with(trial.lab, hardware=True)


def test_dry_run_starts_no_model_or_robot(monkeypatch, tmp_path):
    setup_trial(monkeypatch, tmp_path)
    server, inference = MagicMock(), MagicMock()
    monkeypatch.setattr(operator, "model_server", server)
    monkeypatch.setattr(operator, "run_inference", inference)
    monkeypatch.setattr("sys.argv", ["ur5e-pi05-infer", "20260909_01:1000", "--dry-run"])
    assert operator.infer() == 0
    server.assert_not_called()
    inference.assert_not_called()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_execute_forwards_settings_and_cleans_up_after_interrupt(monkeypatch, tmp_path, interrupt):
    setup_trial(monkeypatch, tmp_path)
    cleaned = []

    @contextmanager
    def server(*_):
        try:
            yield 18005, "owned-instance"
        finally:
            cleaned.append(True)

    inference = MagicMock(side_effect=KeyboardInterrupt if interrupt else None)
    monkeypatch.setattr(operator, "model_server", server)
    monkeypatch.setattr(operator, "run_inference", inference)
    monkeypatch.setattr("sys.argv", ["ur5e-pi05-infer", "20260909_01:1000", "--execute"])
    assert operator.infer() == (130 if interrupt else 0)
    args = inference.call_args.args[0]
    assert (args.mode, args.action_steps, args.chunks, args.speed) == ("execute", 20, 30, 0.6)
    assert args.port == 18005 and args.expected_instance_id == "owned-instance"
    assert cleaned == [True]
    result_path = next((tmp_path / "logs/pi05_infer").glob("*/result.json"))
    assert json.loads(result_path.read_text())["status"] == ("interrupted" if interrupt else "completed")


def test_model_start_failure_cleans_only_owned_process_without_robot_access(monkeypatch, tmp_path):
    setup_trial(monkeypatch, tmp_path)
    process = MagicMock()
    process.poll.return_value = None
    popen = MagicMock(return_value=process)
    inference = MagicMock()
    monkeypatch.setattr(operator.subprocess, "Popen", popen)
    monkeypatch.setattr(operator, "free_loopback_port", lambda: 18005)
    monkeypatch.setattr(operator, "wait_ready", MagicMock(side_effect=RuntimeError("warmup failed")))
    monkeypatch.setattr(operator, "run_inference", inference)
    monkeypatch.setattr("sys.argv", ["ur5e-pi05-infer", "20260909_01:1000", "--execute"])
    assert operator.infer() == 2
    command = popen.call_args.args[0]
    assert command[0] == str(tmp_path / "python")
    assert command[command.index("--diffusion-steps") + 1] == "10"
    assert "--instance-id" in command
    process.terminate.assert_called_once()
    process.kill.assert_not_called()
    inference.assert_not_called()


def test_owned_process_escalates_cleanup_only_after_timeout():
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("owned-model", 5), 0]
    operator.stop_owned_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_readiness_rejects_wrong_model_instance(monkeypatch):
    process = MagicMock()
    process.poll.return_value = None
    client = MagicMock()
    client.__enter__.return_value.metadata = {"instance_id": "other", "checkpoint_path": "checkpoint"}
    monkeypatch.setattr(operator, "PolicyClient", lambda *_, **__: client)
    trial = SimpleNamespace(contract={}, checkpoint=Path("checkpoint"))
    with pytest.raises(RuntimeError, match="different model"):
        operator.wait_ready(process, trial, 18005, "expected", 1)
