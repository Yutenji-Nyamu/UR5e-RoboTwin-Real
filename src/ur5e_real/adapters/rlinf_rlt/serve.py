"""Frozen PyTorch token/head service over the existing local policy protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .actions import ActionCodec, proprio
from .cache import load_cache, validate_features
from .checkpoint import load
from .models import heads
from .run import load_run
from .storage import read_json, digest
from .train import load_encoder


class RLTPolicy:
    def __init__(self, run, cache, spec, feature_client, encoder, actor, *, device="cuda"):
        self.run, self.cache, self.spec = run, cache, spec
        self.client, self.encoder, self.actor = feature_client, encoder, actor
        self.device, self.calls = device, 0
        self.codec = ActionCodec.from_dict(cache["codec"])
        self.encoder.eval().requires_grad_(False)
        self.actor.eval().requires_grad_(False)

    @torch.no_grad()
    def infer(self, observation):
        started = time.monotonic()
        obs = dict(observation)
        aux = np.asarray(obs.pop("executor_aux"), dtype=np.float32)
        observed_at = obs.pop("rlt_observed_at", None)
        obs.pop("rlt_controller_time_s", None)
        obs.pop("rlt_state_host_time_s", None)
        if aux.shape != (6,) or not np.isfinite(aux).all() or (aux < 0).any() or (aux > 1).any():
            raise ValueError("live executor state must contain six observed values in [0,1]")
        result = self.client.infer(obs)
        features = result["rlt_features"]
        prefix, mask = validate_features(features, self.run["config"]["token"])
        z = self.encoder(
            torch.as_tensor(prefix[None], device=self.device), torch.as_tensor(mask[None], device=self.device)
        ).flatten(1)
        p = proprio(features["normalized_state"], aux)
        reference = self.codec.encode(features["raw_reference"])[: self.run["config"]["real"]["action_steps"]]
        if self.spec["kind"] == "reference":
            action = reference.copy()
        else:
            action = (
                self.actor(
                    z,
                    torch.as_tensor(p[None], device=self.device),
                    torch.as_tensor(reference[None], device=self.device),
                    deterministic=self.spec["kind"] != "online",
                )[0]
                .cpu()
                .numpy()
            )
        physical = self.codec.decode(action, np.asarray(obs["state"])[:6])
        actions = np.asarray(result["actions"]).copy()
        k = len(action)
        actions[:k, :6] = physical[:, :6]
        actions[:k, 7:13] = physical[:, :6]
        actions[:k, 13] = physical[:, 6]
        self.calls += 1
        return {
            "actions": actions,
            "rlt": {
                "z": z[0].cpu().numpy(),
                "proprio": p,
                "reference": reference,
                "action": action,
                "call": self.calls,
                "observation_age_s": time.monotonic() - observed_at if observed_at is not None else None,
            },
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--round", required=True, type=Path)
    parser.add_argument("--feature-port", required=True, type=int)
    parser.add_argument("--feature-instance", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    directory, run, _ = load_run(args.run)
    cache = load_cache(directory)
    spec = read_json(args.round / "spec.json")
    if spec["run_id"] != run["run_id"]:
        raise ValueError("round belongs to another run")
    torch.manual_seed(spec["seed"])
    encoder = load_encoder(directory, run, cache, spec["token"], args.device)
    saved = load(spec["heads"], run, cache, kind="heads", token_sha256=spec["token"]["sha256"])
    actor, _ = heads(run["config"])
    actor.load_state_dict(saved["learner"]["actor"])
    actor.to(args.device)
    from ..robotwin_pi05.client import PolicyClient
    from ..robotwin_pi05.dataset import offline_observation
    from ..robotwin_pi05.native import add_native_paths

    with PolicyClient(
        run["contract"], port=args.feature_port, timeout_s=run["config"]["real"]["rpc_timeout_s"]
    ) as client:
        if client.metadata.get("instance_id") != args.feature_instance:
            raise ValueError("feature service identity changed")
        policy = RLTPolicy(run, cache, spec, client, encoder, actor, device=args.device)
        obs, _ = offline_observation(Path(run["dataset"]))
        obs["executor_aux"] = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        policy.infer(obs)
        add_native_paths(model=False)
        # This server has no model/JAX dependency; import just its serving module.
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        metadata = {
            "ur5e_contract": run["contract"],
            "checkpoint_path": run["checkpoint"],
            "training_status": client.metadata["training_status"],
            "instance_id": args.instance_id,
            "run_id": run["run_id"],
            "round_id": spec["round_id"],
            "spec_sha256": digest(spec),
            "token_sha256": spec["token"]["sha256"],
            "head_sha256": spec["heads"]["sha256"],
            "kind": spec["kind"],
            "seed": spec["seed"],
            "cache_id": cache["cache_id"],
        }
        print(f"[RLT HEAD READY] port={args.port} kind={spec['kind']}", flush=True)
        WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
