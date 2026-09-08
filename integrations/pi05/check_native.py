"""Run with .venv/pi05/bin/python; real LeRobot and native transform checks, no robot."""

import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from ur5e_real.adapters.robotwin_pi05.config import make_config
from ur5e_real.adapters.robotwin_pi05.dataset import offline_observation, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    contract, ready = validate_dataset(args.dataset)
    config = make_config(args.dataset)
    from openpi import transforms
    from openpi.training import data_loader
    from openpi.shared import download

    filesystem = MagicMock()
    filesystem.info.return_value = {"type": "file", "size": 1}
    filesystem.get.side_effect = OSError("injected transfer failure")
    with tempfile.TemporaryDirectory(prefix="ur5e-pi05-download-check-") as directory:
        with patch("fsspec.core.url_to_fs", return_value=(filesystem, "unused")):
            try:
                download._download_fsspec("gs://test/fixture", Path(directory) / "partial")
            except OSError as exc:
                assert str(exc) == "injected transfer failure"
            else:
                raise AssertionError("native downloader swallowed a transfer failure")

    data = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(data, 50, config.model)
    assert len(raw) == ready["transitions"]
    training_input = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])
    inference_input = transforms.compose(data.data_transforms.inputs)
    normalize = transforms.Normalize(data.norm_stats, use_quantiles=True)
    unnormalize = transforms.Unnormalize(data.norm_stats, use_quantiles=True)
    output = transforms.compose(data.data_transforms.outputs)
    offset = 0
    for run in contract["run_ids"]:
        with np.load(args.dataset / "ur5e_adapter" / f"{run}.npz", allow_pickle=False) as episode:
            length = len(episode["vectors"]) - 1
        for i in (0, length // 2, length - 1):
            obs, absolute = offline_observation(args.dataset, run, i)
            from_dataset = training_input(raw[offset + i])
            from_rpc = inference_input(obs | {"actions": absolute.copy()})
            for key in ("state", "actions"):
                np.testing.assert_allclose(from_dataset[key], from_rpc[key], atol=1e-6)
            for key in from_rpc["image"]:
                np.testing.assert_array_equal(from_dataset["image"][key], from_rpc["image"][key])
            assert not from_rpc["image_mask"]["left_wrist_0_rgb"]
            normalized = normalize(from_rpc)
            padded = transforms.PadStatesAndActions(32)(normalized)
            restored = output(unnormalize({"state": padded["state"].copy(), "actions": padded["actions"].copy()}))
            np.testing.assert_allclose(restored["actions"], absolute, atol=2e-6)
        offset += length
    transformed = data_loader.transform_dataset(raw, data)
    sample = transformed[0]
    assert sample["actions"].shape == (50, 32) and sample["state"].shape == (32,)
    assert all(img.shape == (224, 224, 3) for img in sample["image"].values())
    assert np.isfinite(sample["actions"]).all()
    print(
        json.dumps(
            {
                "native_loader": "passed",
                "download_error_propagation": "passed",
                "train_rpc_rgb_parity": "passed",
                "native_quantile_delta_inverse": "passed",
                "transitions": len(raw),
                "action_shape": list(sample["actions"].shape),
                "mask": "left wrist false",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
