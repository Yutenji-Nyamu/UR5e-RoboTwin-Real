"""Frozen native JAX prefix/reference provider; ordinary pi05 inference is unchanged.

Sampling is adapted from RoboTwin@2107203 policy/pi05/src/openpi/models/pi0.py
(OpenPI, Apache-2.0). The only sampling change is retaining final prefix states.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from types import MethodType

import numpy as np

from .native import add_native_paths
from .dataset import offline_observation
from .serve import load_policy
from ..rlinf_rlt.config import FEATURE_VERSION


def sample_with_features(self, rng, observation, *, num_steps=10, noise=None):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    observation = model_lib.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_hidden, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time_value = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time_value, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_visibility = jnp.broadcast_to(
            prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1])
        )
        mask = jnp.concatenate([prefix_visibility, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * velocity, time_value + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    # Native prefix order is images followed by the language/discretized-state prompt.
    prompt_length = 0 if observation.tokenized_prompt is None else observation.tokenized_prompt.shape[-1]
    image_length = prefix_tokens.shape[1] - prompt_length
    return actions, prefix_hidden[:, :image_length], prefix_mask[:, :image_length]


class FrozenFeaturePolicy:
    def __init__(self, policy):
        import jax
        from openpi.shared.nnx_utils import module_jit

        if policy._is_pytorch_model:
            raise ValueError("the successful baseline is the native JAX checkpoint")
        self.policy = policy
        self.rng = jax.random.key(0)
        self.sample = module_jit(MethodType(sample_with_features, policy._model))

    def infer(self, observation, *, noise=None):
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation

        started = time.monotonic()
        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, observation))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        self.rng, key = jax.random.split(self.rng)
        kwargs = dict(self.policy._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            kwargs["noise"] = noise[None] if noise.ndim == 2 else noise
        raw, prefix, mask = self.sample(key, Observation.from_dict(batch), **kwargs)
        raw, prefix, mask = np.asarray(raw[0]), np.asarray(prefix[0], dtype=np.float32), np.asarray(mask[0])
        normalized_state = np.asarray(batch["state"][0], dtype=np.float32)
        decoded = self.policy._output_transform({"state": normalized_state.copy(), "actions": raw.copy()})
        return {
            **decoded,
            "rlt_features": {
                "version": FEATURE_VERSION,
                "prefix": prefix,
                "mask": mask.astype(bool),
                "normalized_state": normalized_state,
                "raw_reference": raw.astype(np.float32),
            },
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    add_native_paths()
    policy, contract, verification = load_policy(args.dataset, args.checkpoint, diffusion_steps=10)
    features = FrozenFeaturePolicy(policy)
    observation, _ = offline_observation(args.dataset)
    # Same observation/noise parity before publishing the feature service.
    noise = np.random.default_rng(0).normal(size=(50, 32)).astype(np.float32)
    expected = policy.infer(observation, noise=noise)["actions"]
    sample = features.infer(observation, noise=noise)
    error = float(np.max(np.abs(sample["actions"] - expected)))
    if not np.isfinite(error) or error > 1e-4:
        raise RuntimeError(f"native feature/reference parity failed: {error}")
    features.infer(observation)  # also compile the normal RNG path before hardware can connect
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    metadata = {
        "ur5e_contract": contract,
        "training_status": verification["status"],
        "instance_id": args.instance_id,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "rlt_feature_version": FEATURE_VERSION,
        "parity_max_abs": error,
        "prefix_shape": list(sample["rlt_features"]["prefix"].shape),
        "diffusion_steps": 10,
    }
    print(f"[RLT FEATURES READY] port={args.port}, parity={error:.3g}", flush=True)
    WebsocketPolicyServer(features, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
