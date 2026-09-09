"""Development-only native dummy-model plumbing test; NEVER a pi05_base SFT result."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ur5e_real.adapters.robotwin_pi05 import serve, train
from ur5e_real.adapters.robotwin_pi05.config import make_config, action_expert_path
from ur5e_real.adapters.robotwin_pi05.contract import decode_actions
from ur5e_real.adapters.robotwin_pi05.dataset import offline_observation, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exp-name", default="dummy_plumbing_01")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    options = SimpleNamespace(
        dataset=args.dataset,
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=1,
        warmup_steps=0,
        learning_rate=2.5e-5,
        resume=args.resume,
        checkpoint_base=Path(".venv/pi05-smoke"),
        params="TEST_RANDOM_NO_BASE_WEIGHTS",
        schedule_steps=10,
        save_interval=2,
        eval_interval=2,
        eval_points=1,
    )
    full_config = make_config(
        args.dataset,
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=1,
        warmup_steps=0,
        checkpoint_base=options.checkpoint_base,
        schedule_steps=10,
        save_interval=2,
    )
    from flax import nnx, traverse_util
    import jax
    from ur5e_real.adapters.robotwin_pi05.training_model import Pi05TrainingConfig
    from openpi.training.weight_loaders import NoOpWeightLoader

    # Count the actual full-sized architecture without allocating/restoring its full weights.
    abstract = nnx.eval_shape(full_config.model.create, jax.random.key(0))
    flat = traverse_util.flatten_dict(nnx.state(abstract, nnx.Param).to_pure_dict())
    full_counts = {
        group: sum(
            int(value.size) for path, value in flat.items() if action_expert_path(path) == (group == "trainable")
        )
        for group in ("trainable", "frozen")
    }
    print("[FULL ARCHITECTURE SHAPES ONLY] " + json.dumps(full_counts), flush=True)
    # Native dummy Gemma variants, real native vision trunk/flow loss/optimizer/checkpoint.
    tiny = replace(
        full_config,
        name="pi05_NATIVE_DUMMY_TEST_ONLY",
        model=Pi05TrainingConfig(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=NoOpWeightLoader(),
    )
    with patch.object(train, "make_config", return_value=tiny):
        train.run_training(options)
    checkpoint = tiny.checkpoint_dir / str(args.steps)
    with patch.object(serve, "make_config", return_value=tiny):
        policy, _, _ = serve.load_policy(args.dataset, checkpoint, diffusion_steps=2)
    observation, _ = offline_observation(args.dataset)
    actions = policy.infer(observation)["actions"]
    decode_actions(actions)
    assert np.isfinite(actions).all()
    report = {
        "scope": "random native dummy-model plumbing ONLY, not pi05_base training or robot validation",
        "full_architecture_parameter_counts": full_counts,
        "dummy_backward_optimizer_checkpoint_reload_inference": "passed",
        "actions_shape": list(actions.shape),
    }
    write_json(checkpoint / "dummy_plumbing_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
