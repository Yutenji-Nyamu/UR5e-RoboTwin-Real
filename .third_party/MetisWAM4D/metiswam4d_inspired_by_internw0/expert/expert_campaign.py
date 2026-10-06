"""RoboDojo collection campaign with the privileged scripted expert: the ``metiswam4d.eval.rdj_campaign`` manager
(rounds, claims, resume manifests, watchdog with keep-alive restart, summary) with no policy server and the Isaac
clients running ``expert.expert_client``.

    prepare   rdj_campaign.prepare (per-step observations, recording, uncapped episodes) + expert fields
    run       N Isaac clients per GPU
    client    one client in the foreground for one task config (development; ``--reload`` / ``--probe``)
    summary   rdj_campaign.summary
    status    done / successful episodes of every campaign directory under ``--root``

Per-episode expert outcomes (steps, wall time, notes) are appended to ``<out>/logs/expert_episodes.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from metiswam4d.eval import rdj_campaign as base

POLICY_NAME = "ScriptedExpert"
_client_command = base.client_command
_client_env = base.client_env


def client_command(port: int, campaign: dict, out: Path, variant: dict) -> list[str]:
    command = _client_command(port, campaign, out, variant)
    command[command.index("metiswam4d.eval.rdj_client")] = "metiswam4d_inspired_by_internw0.expert.expert_client"
    command[command.index("--policy_name") + 1] = campaign["policy_name"]
    return command


def client_env(gpu: int, campaign: dict, out: Path, variant: dict, target: int) -> dict:
    env = _client_env(gpu, campaign, out, variant, target)
    env.update(METIS_POLICY_NAME=campaign["policy_name"], METIS_EXPERT_LOG=str(out / "logs" / "expert_episodes.jsonl"),
               METIS_RECORD_4D="1" if campaign.get("record_4d") else "0", PYTHONDONTWRITEBYTECODE="1")
    return env


class NoServer:
    pid, returncode = None, None

    def poll(self):
        return None


class Manager(base.Manager):
    def start_server(self, gpu: int):
        return NoServer()


def prepare(args) -> dict:
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = base.prepare(out, "scripted_expert", "scripted_expert", "scripted_expert", args.episodes_per_round, 1,
                            args.clients_per_gpu, {}, args.info, args.only, args.episodes, args.video_episodes, 0, None,
                            args.layout_offset, args.record, args.env_seed, True)
    manifest.update(policy_name=POLICY_NAME, backend="expert", record_4d=bool(args.record_4d))
    manifest["protocol"].update(
        rounds=None, stop_after="privileged scripted expert (ground-truth object poses + cuRobo), one command per step",
        policy_seed=None, episode_seed="deterministic expert", geometry="none", rgb="raw 640x480 Isaac RGB")
    base.write_json(out / "campaign.json", manifest)
    return manifest


def run_client(out: Path, campaign: dict, name: str, gpu: int, target: int | None, reload: bool, probe: bool) -> int:
    variant = next(v for v in campaign["variants"] if v["name"] == name)
    path = base.result_file(out, variant)
    done = len(base.read_details(out, variant))
    target = target or variant["episodes"]
    if path is not None or campaign["protocol"].get("layout_offset", 0):
        base.synthesize_resume_manifest(path, campaign, variant, campaign["protocol"].get("layout_offset", 0),
                                        base.save_dir(out, campaign, variant))
    env = client_env(gpu, campaign, out, variant, target)
    env.update(METIS_EXPERT_RELOAD="1" if reload else "0", METIS_EXPERT_PROBE="1" if probe else "0")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    print(f"{name}: episodes {done} -> {target} on gpu {gpu}", flush=True)
    return subprocess.call(client_command(0, campaign, out, variant), cwd=base.PROJECT, env=env)


def status(root: Path) -> None:
    """Finished / successful episodes and mean seconds per campaign directory under ``root``."""
    for path in sorted(root.glob("*/campaign.json")):
        out = path.parent
        campaign = json.loads(path.read_text())
        details = [d for v in campaign["variants"] for d in base.read_details(out, v)]
        target = sum(v["episodes"] for v in campaign["variants"])
        log = out / "logs" / "expert_episodes.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        seconds = sum(r["seconds"] for r in rows) / len(rows) if rows else float("nan")
        steps = [r["steps"] for r in rows if r["success"]]
        print(f"{out.name:48s} {len(details):3d}/{target:3d} done  {sum(bool(d['success']) for d in details):3d} success  "
              f"{seconds:6.1f} s/episode  success steps {min(steps, default=0)}-{max(steps, default=0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes-per-round", type=int, default=5)
    p.add_argument("--clients-per-gpu", type=int, default=2)
    p.add_argument("--info", default="expert")
    p.add_argument("--only", action="append")
    p.add_argument("--episodes", type=int)
    p.add_argument("--video-episodes", type=int, default=2)
    p.add_argument("--layout-offset", type=int, default=0)
    p.add_argument("--record")
    p.add_argument("--record-4d", action="store_true")
    p.add_argument("--env-seed", type=int, required=True, choices=(1, 2))
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--gpus", required=True)
    r.add_argument("--clients", type=int)
    r.add_argument("--stall-seconds", type=int, default=1800)
    r.add_argument("--keepalive-pid-file")
    r.add_argument("--min-free-mb", type=int, default=12000)
    c = sub.add_parser("client")
    c.add_argument("--output", required=True)
    c.add_argument("--variant", required=True)
    c.add_argument("--gpu", type=int, default=0)
    c.add_argument("--target", type=int)
    c.add_argument("--reload", action="store_true")
    c.add_argument("--probe", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--output", required=True)
    t = sub.add_parser("status")
    t.add_argument("--root", default="/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/scripted_expert")
    args = ap.parse_args()
    if args.command == "status":
        status(Path(args.root))
        return
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
    if args.command == "client":
        sys.exit(run_client(out, campaign, args.variant, args.gpu, args.target, args.reload, args.probe))
    base.client_command = client_command
    base.client_env = client_env
    manager = Manager(out, [int(g) for g in args.gpus.split(",")], args.clients or campaign["clients_per_gpu"], 0,
                      args.stall_seconds, args.keepalive_pid_file, args.min_free_mb)
    sys.exit(manager.run())


if __name__ == "__main__":
    main()
