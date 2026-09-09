import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ur5e_real.adapters.robotwin_pi05.config import make_config
from ur5e_real.adapters.robotwin_pi05.evaluate import action_metrics, summarize
from ur5e_real.adapters.robotwin_pi05.telemetry import ResourceMonitor, parse_gpu_csv
from ur5e_real.adapters.robotwin_pi05.train import training_recipe


def test_metrics_exclude_terminal_padding_and_unused_dimensions():
    target = np.zeros((50, 14))
    prediction = target.copy()
    prediction[3:, :6] = 100  # 47 padded targets must not dominate the metric.
    prediction[:, 7:13] = 100  # The copy/dummy dimensions are not physical axes.
    prediction[:3, :6] = 0.01
    prediction[:3, 13] = 0.8
    target[:3, 13] = 1
    report = action_metrics(prediction, target, np.arange(50) < 3)
    assert report["valid_targets"] == 3
    assert report["padded_targets_excluded"] == 47
    assert report["joint_mae_rad"] == pytest.approx(0.01)
    assert report["prefix_joint_mae_rad"] == pytest.approx(0.01)
    assert report["gripper_accuracy"] == 1
    assert report["gripper_mae"] == pytest.approx(0.2)
    assert summarize([report])["joint_mae_rad"] == pytest.approx(0.01)


def test_empty_masks_and_nonfinite_predictions_rejected():
    target = np.zeros((50, 14))
    with pytest.raises(ValueError):
        action_metrics(target, target, np.zeros(50, dtype=bool))
    with pytest.raises(ValueError):
        action_metrics(target + np.nan, target, np.ones(50, dtype=bool))


def test_gpu_na_is_unknown_not_zero():
    parsed = parse_gpu_csv("0, GPU-test, 49140, 12000, 36000, 80, [N/A], 65\n")
    assert parsed[0]["memory_free_mib"] == 36000
    assert parsed[0]["power_w"] is None
    assert parse_gpu_csv("0, gpu, 1, 0, 1, NaN, 10, 40")[0]["utilization_pct"] is None
    with pytest.raises(ValueError):
        parse_gpu_csv("broken,row")


def test_monitor_rejects_nonfinite_intervals(tmp_path):
    for interval in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError):
            ResourceMonitor(tmp_path, interval_s=interval)


def test_monitor_records_phases_peaks_and_errors(tmp_path):
    samples = iter(
        [
            {
                "gpus": [{"uuid": "gpu", "memory_used_mib": 100, "memory_free_mib": 900}],
                "host": {"available_bytes": 1000, "process_rss_bytes": 20},
                "errors": [],
            },
            {
                "gpus": [{"uuid": "gpu", "memory_used_mib": 200, "memory_free_mib": 800}],
                "host": {"available_bytes": 900, "process_rss_bytes": 30},
                "errors": ["GPU metric unavailable"],
            },
            {"gpus": [], "host": {}, "errors": []},
        ]
    )
    with ResourceMonitor(tmp_path, interval_s=60, sampler=lambda: next(samples)) as monitor:
        monitor.phase("train", 3)
        monitor.sample()
    summary = json.loads((tmp_path / "resources_summary.json").read_text())
    assert summary["samples"] == 3
    assert summary["gpus"]["gpu"]["memory_used_mib"] == 200
    assert summary["gpus"]["gpu"]["memory_free_mib"] == 800
    assert summary["host"]["available_bytes"] == 900
    rows = [json.loads(row) for row in (tmp_path / "resources.jsonl").read_text().splitlines()]
    assert rows[1]["phase"] == "train" and rows[1]["step"] == 3
    assert summary["sampled_peaks_only"] is True


def test_sampler_failure_does_not_abort_training(tmp_path):
    with ResourceMonitor(tmp_path, interval_s=60, sampler=lambda: 1 / 0):
        pass
    assert json.loads((tmp_path / "resources_summary.json").read_text())["errors"]


def test_batch_and_schedule_rejected_before_loading_native_runtime():
    for options in ({"batch_size": 0}, {"batch_size": 129}, {"schedule_steps": 10}, {"num_workers": 9}):
        with pytest.raises(ValueError):
            make_config(Path("unused"), **options)
    with patch("ur5e_real.adapters.robotwin_pi05.config.validate_dataset", return_value=({}, {"transitions": 3})):
        with pytest.raises(ValueError, match="drop_last"):
            make_config(Path("unused"), batch_size=4)


def test_budget_extension_preserves_recipe_but_batch_change_does_not():
    options = SimpleNamespace(
        params="base",
        steps=1000,
        batch_size=8,
        warmup_steps=100,
        learning_rate=2.5e-5,
        image_augmentation=False,
        num_workers=0,
    )
    config = SimpleNamespace(lr_schedule=SimpleNamespace(decay_steps=3000), seed=42)
    original = training_recipe(options, config, {}, {})
    options.steps = 2000
    assert training_recipe(options, config, {}, {}) == original
    options.batch_size = 4
    assert training_recipe(options, config, {}, {}) != original
