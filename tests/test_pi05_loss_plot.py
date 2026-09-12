import importlib.util
import json

import pytest

from ur5e_real.adapters.robotwin_pi05.native import REPOSITORY


@pytest.mark.parametrize("requested_steps", [1000, 1500, None])
def test_loss_plot_uses_run_budget(tmp_path, monkeypatch, requested_steps):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    spec = importlib.util.spec_from_file_location("pi05_loss_plot_test", REPOSITORY / "scripts/plot_pi05_loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"step": step, "loss": 0.4 / step, "time_utc": "2026-09-12T07:00:00+00:00"}) + "\n"
            for step in range(1, 4)
        )
    )
    if requested_steps is not None:
        (tmp_path / "invocation.json").write_text(json.dumps({"requested_steps": requested_steps}))
    output = tmp_path / "loss.png"
    monkeypatch.setattr("sys.argv", ["plot", "--metrics", str(metrics), "--output", str(output), "--window", "2"])
    module.main()
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["step"] == 3
    assert summary["total_steps"] == (requested_steps or 1000)
    assert output.is_file()
