"""Offline fixed-noise parity: old native inference versus new frozen RLT features.

Run with .venv/pi05/bin/python. Default uses a native dummy model. --checkpoint
selects an audited local SFT checkpoint. Neither mode connects hardware or trains.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="optional existing local checkpoint reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from ur5e_real.adapters.robotwin_pi05.native import add_native_paths

    add_native_paths()
    import jax
    import jax.numpy as jnp
    from ur5e_real.adapters.robotwin_pi05.rlt_features import FrozenFeaturePolicy

    if args.checkpoint:
        from ur5e_real.pi05_operator import load_trial
        from ur5e_real.adapters.robotwin_pi05.serve import load_policy
        from ur5e_real.adapters.robotwin_pi05.dataset import offline_observation

        trial = load_trial(args.checkpoint)
        policy, _, _ = load_policy(trial.dataset, trial.checkpoint, diffusion_steps=10)
        observation, _ = offline_observation(trial.dataset)
        source = str(trial.checkpoint)
    else:
        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy

        config = Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=50, max_token_len=16
        )
        model = config.create(jax.random.key(0))
        fake = config.fake_obs()
        masks = dict(fake.image_masks)
        masks["left_wrist_0_rgb"] = jnp.zeros((1,), dtype=bool)
        fake = dataclasses.replace(fake, image_masks=masks)
        observation = jax.tree.map(lambda x: np.asarray(x[0]), fake.to_dict())
        policy = Policy(model, sample_kwargs={"num_steps": 10})
        source = "native_dummy_pi05"
    features = FrozenFeaturePolicy(policy)
    noise = np.random.default_rng(0).normal(size=(50, 32)).astype(np.float32)
    expected = policy.infer(observation, noise=noise)
    actual = features.infer(observation, noise=noise)
    error = float(np.abs(expected["actions"] - actual["actions"]).max())
    if not np.isfinite(error) or error > 1e-4:
        raise AssertionError(f"native reference parity error: {error}")
    feature = actual["rlt_features"]
    if feature["prefix"].shape[0] != 768 or int(feature["mask"].sum()) != 512:
        raise AssertionError("expected three image slots, only head/right-wrist valid")
    features.infer(observation)  # compile RNG path separately
    started = time.monotonic()
    live = features.infer(observation)
    report = {
        "source": source,
        "physical_execution": False,
        "training": False,
        "native_max_abs_error": error,
        "prefix_shape": list(feature["prefix"].shape),
        "valid_image_tokens": int(feature["mask"].sum()),
        "raw_reference_shape": list(feature["raw_reference"].shape),
        "warm_feature_wall_s": time.monotonic() - started,
        "live_finite": bool(np.isfinite(live["actions"]).all()),
    }
    if args.output:
        from ur5e_real.adapters.rlinf_rlt.storage import atomic_json

        atomic_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
