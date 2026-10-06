"""InternW0-Delta policy server for the RoboDojo campaign manager (runs in the InternW0 Python 3.11 env).

One model per GPU, one InternW0 session per simulator episode (``session`` id chosen by the client), so several
Isaac clients can share a server without mixing memory frames or pending actions.  Each ``act`` call carries the
acknowledgements of the actions executed since the previous call — one entry per action, the observation taken
after it or ``None`` when InternW0's memory does not use that frame — the last entry being the current observation.
This reproduces the official deploy (``update_obs`` after every action, ``get_action`` at every re-plan).

    <internw0 python> -m metiswam4d_inspired_by_internw0.eval.iw0_server --port P --output DIR
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROBODOJO = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo")
POLICY_DIR = Path("/m2v_intern_v3/danglingwei/files/InternW0_delta_src/XPolicyLab/policy/InternW0_delta")
ZOO = Path("/ytech_milm_intern/danglingwei/model_zoos/InternW0-Delta")
DEFAULTS = {
    "checkpoint_path": str(ZOO / "InternW0-Delta-RoboDojo/robodojo.pt"),
    "vlm_model_path": str(ZOO / "RynnBrain1.1-2B"),
    "base_model_dir": "/m2v_intern/_public_models/Wan-AI/Wan2.2-TI2V-5B",
    "dataset_stats_path": str(POLICY_DIR / "config/dataset_stats.json"),
    "train_config_path": str(POLICY_DIR / "config/eval_model.yaml"),
    "wam_root": str(POLICY_DIR / "wam_runtime"),
}


def _import_internw0():
    """``XPolicyLab.policy.InternW0_delta`` from the sparse checkout, on top of the local XPolicyLab package."""
    import importlib.util
    sys.path[:0] = [str(ROBODOJO), str(ROBODOJO / "XPolicyLab")]
    import XPolicyLab.policy as policy_pkg
    name = "XPolicyLab.policy.InternW0_delta"
    spec = importlib.util.spec_from_file_location(name, POLICY_DIR / "__init__.py",
                                                  submodule_search_locations=[str(POLICY_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    policy_pkg.InternW0_delta = module
    from XPolicyLab.policy.InternW0_delta import model as model_module
    from XPolicyLab.policy.InternW0_delta._adapter_base import instruction
    from XPolicyLab.utils.process_data import unpack_robot_state
    return model_module.Model, instruction, unpack_robot_state


class InternW0Server:
    def __init__(self, output: Path, overrides: dict | None = None, log=print):
        import yaml
        Model, self._instruction, self._unpack = _import_internw0()
        cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
        cfg.update(DEFAULTS)
        cfg.update(overrides or {})
        cfg.update(env_cfg_type="arx_x5", action_type="joint", bench_name="RoboDojo")
        self.model = Model(cfg)
        self.sessions: dict[str, object] = {}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.calls = 0

    def reset(self):
        return {"ok": True}

    def begin(self, payload: dict):
        """New episode for ``payload["session"]``; returns the observation cadence the client must follow."""
        self.sessions[str(payload["session"])] = self.model.session_factory()
        return self.info()

    def recent_capture_mod(self) -> int:
        """Steps whose post-action frame InternW0's memory keeps (besides re-plan frames)."""
        return (-self.model.action_horizon) % self.model.replan_steps

    def act(self, payload: dict) -> dict:
        """``payload = {"session", "acks": [obs | None, ...], "instruction"?}``."""
        session, acks = str(payload["session"]), list(payload["acks"])
        instruction_text = payload.get("instruction")
        state = self.sessions.get(session)
        if state is None:
            state = self.sessions[session] = self.model.session_factory()
        if not acks or acks[-1] is None:
            raise ValueError("the last acknowledgement must carry the current observation")
        current = None
        for obs in acks:
            adapted = self.model._adapt_obs(obs) if obs is not None else None
            if state.pending_model_actions:
                state.update_obs(adapted)
            current = adapted if adapted is not None else current
        text = instruction_text or self._instruction(acks[-1], self.model.default_instruction)
        t0 = time.time()
        packed = np.asarray(state.get_action({"observation": current, "instruction": text}), np.float32)
        for gripper in self.model.gripper_slices:
            packed[:, gripper] = np.clip(packed[:, gripper], 0.0, 1.0)
        seconds = time.time() - t0
        self.calls += 1
        if self.calls % 50 == 1:
            self.log(json.dumps({"session": session, "step": int(state.step_count), "seconds": round(seconds, 2),
                                 "instruction": text, "open_sessions": len(self.sessions)}))
        actions = self._unpack(packed, "joint", self.model.robot_action_dim_info, source_type="obs")
        return {"actions": actions, "joint": packed, "step": int(state.step_count), "inference_seconds": seconds}

    def close(self, payload: dict):
        self.sessions.pop(str(payload["session"]), None)
        return {"ok": True}

    def info(self):
        return {"replan_steps": int(self.model.replan_steps), "action_horizon": int(self.model.action_horizon),
                "recent_capture_mod": self.recent_capture_mod()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overrides", default="{}", help="JSON overrides of deploy.yml keys")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    server = InternW0Server(Path(args.output), json.loads(args.overrides), log=lambda m: print(m, flush=True))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    print("MODEL_READY", json.dumps(server.info()), flush=True)
    asyncio.run(PolicyServer(server, PolicyServerConfig(host="127.0.0.1", port=args.port)).serve_forever())


if __name__ == "__main__":
    main()
