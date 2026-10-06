"""RoboDojo campaign with the InternW0-Delta teacher: the ``metiswam4d.eval.rdj_campaign`` manager (protocol, rounds,
claims, resume manifests, watchdog with keep-alive restart, summary) with the InternW0 server / client swapped in.

    prepare   rdj_campaign.prepare + backend fields (policy name, server overrides, 4D recording)
    run       one InternW0 server per GPU (InternW0 Python 3.11 env) + N Isaac clients with ``iw0_client``
    summary   rdj_campaign.summary

Evaluation always uses the official per-step observation cadence; ``--record`` writes the trajectories and
``--record-4d`` adds the head-camera ground truth (depth, instance ids, scene instance poses).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from metiswam4d.eval import rdj_campaign as base

PROJECT = Path(__file__).resolve().parents[2]
IW0_PYTHON = "/ytech_milm_intern/danglingwei/envs/internw0-delta/bin/python"
POLICY_NAME = "InternW0_delta"


def server_env(gpu: int) -> dict:
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONNOUSERSITE", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(PROJECT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", DIFFSYNTH_SKIP_DOWNLOAD="true")
    return env


_rdj_client_command = base.client_command


def client_command(port: int, campaign: dict, out: Path, variant: dict) -> list[str]:
    command = _rdj_client_command(port, campaign, out, variant)
    command[command.index("metiswam4d.eval.rdj_client")] = "metiswam4d_inspired_by_internw0.eval.iw0_client"
    command[command.index("--policy_name") + 1] = campaign["policy_name"]
    return command


SERVERS = {   # backend -> (python, module, env builder)
    "internw0": (IW0_PYTHON, "metiswam4d_inspired_by_internw0.eval.iw0_server", server_env),
    "mem": (base.POLICY_PYTHON, "metiswam4d_inspired_by_internw0.eval.mem_server", base.policy_env),
}


class AttachedServer:
    """A server owned by another manager on this host (``attach_base_port``): sessions are per episode, so several
    campaigns can share one model per GPU.  Alive while its port accepts connections."""

    def __init__(self, port: int):
        self.port, self.pid, self.returncode = port, None, None

    def poll(self):
        import socket as sk
        with sk.socket() as s:
            s.settimeout(2)
            return None if s.connect_ex(("127.0.0.1", self.port)) == 0 else 1


class Manager(base.Manager):
    def start_server(self, gpu: int):
        attach = self.campaign.get("attach_base_port")
        if attach:
            server = AttachedServer(int(attach) + gpu)
            while server.poll() is not None and not self.stop.wait(10):
                pass
            self.log(f"gpu{gpu}: attached to the server on port {server.port}")
            return server
        port = self.base_port + gpu
        self.wait_port_free(port)
        (self.out / f"gpu{gpu}").mkdir(exist_ok=True)
        overrides = json.dumps(self.campaign.get("server_overrides") or {})
        python, module, env_fn = SERVERS[self.campaign.get("backend", "internw0")]
        server = self.launch([python, "-m", module, "--port", str(port), "--output", str(self.out / f"gpu{gpu}"),
                              "--overrides", overrides],
                             env_fn(gpu), self.out / "logs" / f"policy_gpu{gpu}_{self.host}.log")
        from websockets.sync.client import connect
        deadline = time.monotonic() + 1800
        while not self.stop.is_set():
            if server.poll() is not None:
                raise RuntimeError(f"GPU {gpu} InternW0 server exited with {server.returncode}")
            try:
                with connect(f"ws://127.0.0.1:{port}", open_timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"GPU {gpu} InternW0 server not ready after 30 min")
                self.stop.wait(3)
        self.log(f"InternW0 server gpu{gpu} ready on port {port} (pid {server.pid})")
        return server


def prepare(args) -> dict:
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mem = args.backend == "mem"
    execute = args.execute_steps if mem else 10
    manifest = base.prepare(out, args.config if mem else "internw0", args.model_file, args.model_file,
                            args.episodes_per_round, execute, args.clients_per_gpu, {}, args.info, args.only,
                            args.episodes, args.video_episodes, args.policy_seed, None, args.layout_offset, args.record,
                            args.env_seed, True,
                            json.loads(Path(args.exclude_layouts).read_text()) if args.exclude_layouts else None)
    manifest.update(backend=args.backend, record_4d=bool(args.record_4d), attach_base_port=args.attach_base_port)
    if mem:
        manifest.update(policy_name="MetisWAM4D_IW0", server_overrides={
            "config": str(Path(args.config).resolve()), "weights": args.model_file, "execute_steps": execute,
            "seed": args.policy_seed})
        manifest["protocol"].update(
            rounds=10, stop_after="action-only sampling on the clean world (memory + current frame), 10 steps",
            geometry="SAPIEN robot-only depth (mm) + mask from live joint states (training renderer)",
            rgb="640x480 -> 320x240 corpus JPEG path -> InternW0 canvas; memory canvases of steps t-32k and 0")
    else:
        manifest.update(policy_name=POLICY_NAME, server_overrides={"checkpoint_path": args.model_file})
        manifest["protocol"].update(
            rounds=10, stop_after="InternW0-Delta: 32-step chunk, re-plan after 10, 10 denoising steps",
            policy_seed=None, episode_seed="InternW0 runtime default (unseeded)",
            geometry="none (InternW0 reads RGB + joint state)", rgb="raw 640x480 Isaac RGB -> InternW0 training canvas")
    base.write_json(out / "campaign.json", manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=("internw0", "mem"), default="internw0")
    p.add_argument("--config", help="mem backend: training config of the model")
    p.add_argument("--execute-steps", type=int, default=32, help="mem backend: actions executed per re-plan")
    p.add_argument("--policy-seed", type=int, default=42)
    p.add_argument("--model-file", default="/ytech_milm_intern/danglingwei/model_zoos/InternW0-Delta/"
                                           "InternW0-Delta-RoboDojo/robodojo.pt")
    p.add_argument("--episodes-per-round", type=int, default=2)
    p.add_argument("--clients-per-gpu", type=int, default=3)
    p.add_argument("--info", default="iw0")
    p.add_argument("--only", action="append")
    p.add_argument("--episodes", type=int)
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--layout-offset", type=int, default=0)
    p.add_argument("--record")
    p.add_argument("--record-4d", action="store_true")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--exclude-layouts")
    p.add_argument("--attach-base-port", type=int, help="share the servers of another campaign on this host")
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--gpus", required=True)
    r.add_argument("--clients", type=int)
    r.add_argument("--base-port", type=int, default=35380)
    r.add_argument("--stall-seconds", type=int, default=3600)
    r.add_argument("--keepalive-pid-file")
    r.add_argument("--min-free-mb", type=int, default=12000)
    s = sub.add_parser("summary")
    s.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        manifest = prepare(args)
        print(json.dumps({"variants": len(manifest["variants"]),
                          "episodes": sum(v["episodes"] for v in manifest["variants"]), "output": args.output}))
        return
    out = Path(args.output).resolve()
    if args.command == "summary":
        base.summary(out)
        print((out / "summary.md").read_text())
        return
    campaign = json.loads((out / "campaign.json").read_text())
    if campaign.get("record_4d"):
        os.environ["METIS_RECORD_4D"] = "1"
    base.client_command = client_command
    port = int(campaign.get("attach_base_port") or args.base_port)   # clients talk to the shared servers
    manager = Manager(out, [int(g) for g in args.gpus.split(",")], args.clients or campaign["clients_per_gpu"],
                      port, args.stall_seconds, args.keepalive_pid_file, args.min_free_mb)
    sys.exit(manager.run())


if __name__ == "__main__":
    main()
