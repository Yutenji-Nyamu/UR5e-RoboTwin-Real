"""Teacher-forced training-set diagnostics, excluding terminal padding; no hardware I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .config import make_config
from .contract import STATE_LAYOUT, decode_actions, require_same_contract
from .dataset import validate_dataset, write_json
from .native import CHECKPOINT_COMPAT


def action_metrics(predicted, expected, valid_mask, *, first_steps=6):
    # Preserve the original training diagnostic; this is not the live execution K.
    # Deployment now defaults to K=20. A matching diagnostic must request first_steps=20.
    joints, _ = decode_actions(predicted)
    expected, mask = np.asarray(expected), np.asarray(valid_mask, dtype=bool)
    if expected.shape != np.asarray(predicted).shape or mask.shape != (len(joints),) or not mask.any():
        raise ValueError("action metrics require matching targets and a nonempty horizon mask")
    if first_steps < 1:
        raise ValueError("first_steps must be positive")
    first_mask = mask & (np.arange(len(mask)) < first_steps)
    if not first_mask.any():
        raise ValueError("no valid targets in the executed prefix")
    delta = np.abs(joints - expected[:, :6])
    grip = np.asarray(predicted)[:, 13]
    correct = (grip >= 0.5) == (expected[:, 13] >= 0.5)
    return {
        "valid_targets": int(mask.sum()),
        "padded_targets_excluded": int((~mask).sum()),
        "joint_mae_rad": float(delta[mask].mean()),
        "joint_max_error_rad": float(delta[mask].max()),
        "joint_mae_per_axis_rad": delta[mask].mean(axis=0).tolist(),
        "gripper_mae": float(np.abs(grip[mask] - expected[mask, 13]).mean()),
        "gripper_accuracy": float(correct[mask].mean()),
        "prefix_targets": int(first_mask.sum()),
        "prefix_joint_mae_rad": float(delta[first_mask].mean()),
        "prefix_gripper_accuracy": float(correct[first_mask].mean()),
    }


def evaluation_samples(config, *, points_per_episode=3):
    if not 1 <= points_per_episode <= 20:
        raise ValueError("evaluation points per episode must be 1..20")
    from openpi.training.data_loader import create_torch_dataset
    from openpi.transforms import compose

    data_config = config.data.create(config.assets_dirs, config.model)
    raw = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    repack = compose(data_config.repack_transforms.inputs)
    samples = []
    for episode, (start, stop) in enumerate(zip(raw.episode_data_index["from"], raw.episode_data_index["to"])):
        start, stop = int(start), int(stop)
        indices = (
            [start + (stop - start) // 2]
            if points_per_episode == 1
            else np.linspace(start, stop - 1, min(points_per_episode, stop - start), dtype=int).tolist()
        )
        for index in indices:
            item = raw[index]
            obs = repack(item)
            expected = np.asarray(obs.pop("actions"), dtype=np.float32).copy()
            # Retain the native mask before repacking discards auxiliary LeRobot fields.
            mask = ~np.asarray(item["action_is_pad"], dtype=bool)
            samples.append(
                {"episode": episode, "index": index, "observation": obs, "expected": expected, "valid_mask": mask}
            )
    return data_config, samples


def summarize(rows):
    targets = sum(row["valid_targets"] for row in rows)
    prefix = sum(row["prefix_targets"] for row in rows)
    result = {
        "scope": "teacher_forced_training_set_only",
        "physical_execution": False,
        "samples": len(rows),
        "valid_targets": targets,
        "padded_targets_excluded": sum(row["padded_targets_excluded"] for row in rows),
        "rows": rows,
    }
    for key in ("joint_mae_rad", "gripper_mae", "gripper_accuracy"):
        result[key] = sum(row[key] * row["valid_targets"] for row in rows) / targets
    for key in ("prefix_joint_mae_rad", "prefix_gripper_accuracy"):
        result[key] = sum(row[key] * row["prefix_targets"] for row in rows) / prefix
    return result


class StateEvaluator:
    """The jitted sampler receives CURRENT params, never a snapshot captured by Policy.module_jit.

    Shares training buffers and does not construct a second full model on the GPU.
    """

    def __init__(self, config, graphdef, *, points_per_episode=3, diffusion_steps=10):
        from flax import nnx
        import jax
        from openpi import transforms
        from openpi.models.model import Observation

        self.jax, self.Observation = jax, Observation
        data, samples = evaluation_samples(config, points_per_episode=points_per_episode)
        input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(config.policy_metadata["prompt"]),
                *data.data_transforms.inputs,
                transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ]
        )
        self.output_transform = transforms.compose(
            [
                *data.model_transforms.outputs,
                transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.data_transforms.outputs,
            ]
        )
        self.samples = [
            {
                **sample,
                "inputs": input_transform(sample["observation"]),
                "noise": np.random.default_rng(config.seed + sample["index"])
                .normal(size=(1, config.model.action_horizon, config.model.action_dim))
                .astype(np.float32),
            }
            for sample in samples
        ]

        def sample_actions(params, observation, noise):
            model = nnx.merge(graphdef, params)
            model.eval()
            return model.sample_actions(jax.random.key(0), observation, noise=noise, num_steps=diffusion_steps)

        self.sample_actions = jax.jit(sample_actions)

    def __call__(self, params):
        rows = []
        for sample in self.samples:
            inputs = self.jax.tree.map(lambda value: np.asarray(value)[None], sample["inputs"])
            observation = self.Observation.from_dict(inputs)
            actions = np.asarray(self.sample_actions(params, observation, sample["noise"]))[0]
            outputs = self.output_transform({"state": sample["inputs"]["state"].copy(), "actions": actions})
            rows.append(
                {
                    "episode": sample["episode"],
                    "index": sample["index"],
                    **action_metrics(outputs["actions"], sample["expected"], sample["valid_mask"]),
                }
            )
        return summarize(rows)


def evaluate_checkpoint(dataset, checkpoint, *, points_per_episode=3):
    contract, ready = validate_dataset(dataset)
    manifest = json.loads((checkpoint / "ur5e_evaluation_manifest.json").read_text())
    require_same_contract(contract, manifest["contract"])
    if manifest["dataset_ready"] != ready or manifest["native_checkpoint_compat"] != CHECKPOINT_COMPAT:
        raise ValueError("evaluation checkpoint dataset or native runtime differs")
    if int(checkpoint.name) != manifest["step"] or manifest.get("checkpoint_saved") is not True:
        raise ValueError("checkpoint save was not completed")
    norm = checkpoint / "assets" / STATE_LAYOUT / "norm_stats.json"
    if hashlib.sha256(norm.read_bytes()).hexdigest() != ready["norm_sha256"]:
        raise ValueError("checkpoint normalization differs")
    config = make_config(dataset, image_augmentation=manifest["image_augmentation"])
    from openpi.policies.policy_config import create_trained_policy

    policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": 10})
    _, samples = evaluation_samples(config, points_per_episode=points_per_episode)
    rows = []
    for sample in samples:
        noise = np.random.default_rng(config.seed + sample["index"]).normal(size=(50, 32)).astype(np.float32)
        outputs = policy.infer(sample["observation"], noise=noise)
        rows.append(
            {
                "episode": sample["episode"],
                "index": sample["index"],
                **action_metrics(outputs["actions"], sample["expected"], sample["valid_mask"]),
            }
        )
    return {"step": manifest["step"], **summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-episode", type=int, default=3)
    args = parser.parse_args()
    started = time.monotonic()
    report = evaluate_checkpoint(args.dataset, args.checkpoint, points_per_episode=args.points_per_episode)
    report["elapsed_s"] = time.monotonic() - started
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
