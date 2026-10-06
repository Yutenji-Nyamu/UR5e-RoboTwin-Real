"""VLABench closed-loop campaign for an OpenWAM-deployable checkpoint, official OpenWAM-alpha protocol.

Protocol (``OpenWAM_Official_260913/benchmarks/vlabench``): the five frozen evaluation tracks of VLABench
(in-distribution, cross-category, common-sense, semantic-instruction, unseen-texture), every task of each track's
config, 50 episodes per task (fewer where the track ships fewer, e.g. cross-category insert_flower = 10).  Each
(track, task) job is one run of the official client ``single_eval.sh`` against a dedicated policy server
(``scripts/deploy.py`` with ``configs/deploy.yaml``: 10 sync denoising steps, DiT cache, compile, seed 42).
Track score = mean over its tasks; Avg = mean over the five tracks.  track_5 (no frozen episodes) is not part of Avg.

Sub-commands (run from the project root with ``/usr/bin/python3.10 -m metiswam4d.eval.vlabench_campaign``):
    run       start one server per slot (``--servers-per-gpu`` per GPU, one port each; a server never serves two
              simulators), then scan the job list in passes: every pass queues the (track, task) jobs without an
              accepted result and the slots drain the queue.  A job is accepted when the client exits 0, wrote its
              result and episode records, and its log has no policy-side traceback (server/transport failure);
              otherwise its output is moved to ``failed/`` and the next pass runs it again.  Episodes the
              upstream evaluator drops on simulator exceptions (MuJoCo PhysicsError) stay dropped, as officially.
    summary   ``summary.json`` / ``summary.md`` from the accepted jobs

Layout: ``<out>/by_task/<task>/<track>/{openwam/evaluation_result.json, <task>/detail_info.json, metrics.json}``
(the official ``multi_eval.sh`` layout), ``<out>/status/<track>__<task>.json``, ``<out>/failed/``,
``<out>/logs/{servers,jobs}/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import threading
import time

OPENWAM = Path("/m2v_intern_v3/danglingwei/codes/OpenWAM_Official_260913")
BENCH = OPENWAM / "benchmarks" / "vlabench"
VLABENCH = Path("/ytech_milm_intern/danglingwei/files/VLABench")
VLABENCH_PYTHON = "/usr/local/vlabench/venv/bin/python"
SERVER_PYTHON = "/usr/bin/python3.10"
ALPHA_VLABENCH = Path("/ytech_milm_intern/danglingwei/datas/VLABench/OpenWAM-Alpha-Sim-VLABench")
PROJECT = Path(__file__).resolve().parents[2]
CLIENT_STUBS = PROJECT / "scripts" / "vlabench" / "client_stubs"
METIS_CLIENT = PROJECT / "scripts" / "vlabench" / "metis_single_eval.py"
TRACKS = ("track_1_in_distribution", "track_2_cross_category", "track_3_common_sense",
          "track_4_semantic_instruction", "track_6_unseen_texture")
SHORT = {"track_1_in_distribution": "ID", "track_2_cross_category": "Cat", "track_3_common_sense": "CS",
         "track_4_semantic_instruction": "Ins", "track_6_unseen_texture": "Tex"}
# track_5 ships no frozen episodes (the official client loads tracks/track_5_cross_task.json when present); here it holds
# the development set: scene configs of the held-out gen4d episodes (scripts/vlabench/build_dev_track.py), not in Avg
DEV_TRACK = "track_5_cross_task"
SHORT[DEV_TRACK] = "Dev"
METRICS = ("success_rate", "progress_score", "intention_score")
PAPER_ALPHA = {  # OpenWAM paper, OpenWAM-alpha (SR, PS, IS) in %
    "track_1_in_distribution": (83.4, 87.9, 75.2), "track_2_cross_category": (38.1, 45.9, 45.9),
    "track_3_common_sense": (58.0, 64.8, 57.9), "track_4_semantic_instruction": (53.8, 64.5, 66.5),
    "track_6_unseen_texture": (61.4, 72.9, 71.6), "avg": (58.9, 67.2, 63.5),
}
POLICY_FAILURE = re.compile(r"transport\.py|openwam2vlabench_interface\.py|metis_single_eval\.py|"
                            r"ConnectionRefusedError|ConnectionResetError|websockets\.exceptions")
JOB_TIMEOUT_S = 6 * 3600
# The official ``_BoundedPromptEmbedCache`` overrides ``__getitem__``, which ``OrderedDict.popitem`` calls on
# eviction: from the 33rd distinct prompt on, the server answers 500 (KeyError) and the episode is dropped.  The
# cache holds deterministic text-encoder outputs, so a bound that is never reached changes nothing else.
PROMPT_CACHE_OVERRIDE = "optimization.prompt_embed_cache.maxsize=100000"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def stamp() -> str:
    return time.strftime("%F %T")


def log(msg: str) -> None:
    print(f"[{stamp()}] {msg}", flush=True)


def job_list(tracks=TRACKS, n_episodes: int = 50, tasks=None) -> list[dict]:
    jobs = []
    for track in tracks:
        config = json.loads((VLABENCH / "VLABench/configs/evaluation/tracks" / f"{track}.json").read_text())
        for task, episodes in config.items():
            if tasks is None or task in tasks:
                jobs.append({"track": track, "task": task, "budget": min(n_episodes, len(episodes))})
    return jobs


def job_name(job: dict) -> str:
    return f"{job['track']}__{job['task']}"


def job_dir(out: Path, job: dict) -> Path:
    return out / "by_task" / job["task"] / job["track"]


def status_path(out: Path, job: dict) -> Path:
    return out / "status" / f"{job_name(job)}.json"


# ---------------------------------------------------------------------------------------------------------------
# policy servers


class Server:
    """``backend``: ``alpha`` (official ``scripts/deploy.py``) or ``metis`` (``metiswam4d.eval.vlabench_policy`` with
    ``metis`` = {model_file, execute_steps, policy, policy_seed})."""

    def __init__(self, slot: int, gpu: str, port: int, ckpt_dir: Path, ckpt_name: str, out: Path,
                 backend: str = "alpha", metis: dict | None = None):
        self.slot, self.gpu, self.port = slot, gpu, port
        self.ckpt_dir, self.ckpt_name = ckpt_dir, ckpt_name
        self.backend, self.metis = backend, metis or {}
        self.log_dir = out / "logs" / "servers"
        self.proc: subprocess.Popen | None = None
        self.starts = 0

    def start(self) -> None:
        self.stop()
        self.starts += 1
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / (f"slot{self.slot:02d}_gpu{self.gpu}_port{self.port}_"
                                   f"{time.strftime('%Y%m%d_%H%M%S')}_start{self.starts}.log")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, OMP_NUM_THREADS="4",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if self.backend == "metis":
            m = self.metis
            cmd = [SERVER_PYTHON, "-m", "metiswam4d.eval.vlabench_policy", "--checkpoint", m["model_file"],
                   "--port", str(self.port), "--execute-steps", str(m["execute_steps"]),
                   "--policy", json.dumps(m["policy"]), "--policy-seed", str(m["policy_seed"])]
            cwd = PROJECT
            env.update(PYTHONPATH=str(PROJECT), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        else:
            cmd = [SERVER_PYTHON, "scripts/deploy.py", "--ckpt-dir", str(self.ckpt_dir), "--ckpt-name", self.ckpt_name,
                   "--device", "cuda:0", "--port", str(self.port), PROMPT_CACHE_OVERRIDE]
            cwd = OPENWAM
        with open(log_file, "w") as fh:
            self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"slot {self.slot}: server pid {self.proc.pid} gpu {self.gpu} port {self.port} -> {log_file.name}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def listening(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=2):
                return True
        except OSError:
            return False

    def wait_ready(self, timeout_s: float = 1800) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not self.alive():
                return False
            if self.listening():
                return True
            time.sleep(5)
        return False

    def stop(self) -> None:
        if self.alive():
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None

    def ensure(self) -> bool:
        for _ in range(3):
            if self.alive() and self.listening():
                return True
            self.start()
            if self.wait_ready():
                log(f"slot {self.slot}: server ready on port {self.port}")
                return True
            log(f"slot {self.slot}: server failed to come up (start {self.starts})")
        return False


def start_servers(servers: list[Server], gpus: list[str]) -> None:
    """One server per GPU at a time (bounded host-RAM / checkpoint-read peak), the next wave once it is up."""
    waves = [[s for s in servers if s.slot // len(gpus) == w] for w in range(math.ceil(len(servers) / len(gpus)))]
    for wave in waves:
        for s in wave:
            s.start()
        for s in wave:
            if s.wait_ready():
                log(f"slot {s.slot}: server ready on port {s.port}")
            else:
                log(f"slot {s.slot}: server not ready; the slot retries before its first job")


# ---------------------------------------------------------------------------------------------------------------
# jobs


def run_job(out: Path, job: dict, server: Server, pass_idx: int) -> dict:
    """One official ``single_eval.sh`` run; returns the status record (``accepted`` or the failure reason)."""
    folder = job_dir(out, job)
    if folder.exists():
        shutil.rmtree(folder)
    log_file = out / "logs" / "jobs" / f"{job_name(job)}.pass{pass_idx}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VLABENCH_PATH=str(VLABENCH), VLABENCH_PYTHON=VLABENCH_PYTHON,
               # IDC nodes enumerate only the CUDA-visible GPU as an EGL device
               VLABENCH_SAVE_DIR=str(out / "by_task" / job["task"]), MUJOCO_EGL_DEVICE_ID="0",
               CUDA_VISIBLE_DEVICES=server.gpu, OMP_NUM_THREADS="4", PYTHONPATH=str(CLIENT_STUBS))
    if server.backend == "metis":   # single_eval.sh with metis_single_eval.py in place of single_eval.py
        env.update(PYTHONPATH=f"{VLABENCH}:{BENCH}:{CLIENT_STUBS}", VLABENCH_ROOT=str(VLABENCH / "VLABench"),
                   MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", PYTHONUNBUFFERED="1")
        cmd = [VLABENCH_PYTHON, str(METIS_CLIENT), "--config", str(BENCH / "policy_config.yml"),
               "--eval-track", job["track"], "--tasks", job["task"], "--n-episodes", str(job["budget"]),
               "--save-dir", str(out / "by_task" / job["task"]), "--host", "127.0.0.1", "--port", str(server.port)]
    else:
        cmd = ["bash", str(BENCH / "single_eval.sh"), job["task"], job["track"], str(job["budget"]), str(server.port)]
    t0 = time.time()
    with open(log_file, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=OPENWAM, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = proc.wait(timeout=JOB_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = "timeout"
    record = {**job, "pass": pass_idx, "slot": server.slot, "gpu": server.gpu, "port": server.port,
              "exit": code, "wall_s": round(time.time() - t0, 1), "log": str(log_file), "finished": stamp()}
    result_file = folder / "openwam" / "evaluation_result.json"
    detail_file = folder / job["task"] / "detail_info.json"
    text = log_file.read_text(errors="replace")
    reason = None
    if code != 0:
        reason = f"exit {code}"
    elif not result_file.exists() or not detail_file.exists():
        reason = "no result files"
    elif POLICY_FAILURE.search(text):
        reason = "policy-side traceback in log"
    if reason is None:
        infos = json.loads(detail_file.read_text())
        metrics = json.loads(result_file.read_text())[job["task"]]
        intention = [i["intention_score"] for i in infos]
        record.update(records=len(infos), dropped=job["budget"] - len(infos),
                      metrics={k: metrics.get(k) for k in METRICS},
                      intention_nan_episodes=sum(1 for v in intention if v is None or v != v))
        if record["intention_nan_episodes"]:  # get_intention_score fell back to NaN: mean over the scored episodes
            record["metrics"]["intention_score"] = _mean(intention)
        if not infos:
            reason = "no episode records"
    record["accepted"] = reason is None
    record["reason"] = reason
    return record


def slot_loop(out: Path, server: Server, jobs: queue.Queue, pass_idx: int, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        if not server.ensure():
            log(f"slot {server.slot}: server unavailable, {job_name(job)} back to the next pass")
            return
        log(f"slot {server.slot}: start {job_name(job)} ({job['budget']} episodes)")
        record = run_job(out, job, server, pass_idx)
        if record["accepted"]:
            write_json(status_path(out, job), record)
            m = record["metrics"]
            log(f"slot {server.slot}: done {job_name(job)} in {record['wall_s'] / 60:.1f} min  "
                f"SR {m['success_rate']:.3f} PS {m['progress_score']:.3f} IS {m['intention_score']}  "
                f"records {record['records']}/{job['budget']}")
        else:
            failed = out / "failed" / f"{job_name(job)}__pass{pass_idx}_{time.strftime('%Y%m%d_%H%M%S')}"
            if job_dir(out, job).exists():
                failed.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(job_dir(out, job)), str(failed))
            with open(out / "failures.jsonl", "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            log(f"slot {server.slot}: REJECTED {job_name(job)}: {record['reason']} (log {record['log']})")
            if not server.alive() or not server.listening() or record["reason"] == "policy-side traceback in log":
                server.start()
                server.wait_ready()


def pending(out: Path, jobs: list[dict]) -> list[dict]:
    return [j for j in jobs if not status_path(out, j).exists()]


def run(out: Path, ckpt_dir: Path, ckpt_name: str, gpus: list[str], servers_per_gpu: int, port_base: int,
        n_episodes: int, max_passes: int, tracks=TRACKS, tasks=None, backend: str = "alpha",
        metis: dict | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    jobs = job_list(tracks, n_episodes=n_episodes, tasks=tasks)
    manifest = {"protocol": "OpenWAM-alpha official VLABench (5 frozen tracks, all tasks, <=50 episodes)",
                "backend": backend, "metis": metis,
                "checkpoint": metis["model_file"] if backend == "metis" else str(ckpt_dir / ckpt_name),
                "openwam": str(OPENWAM), "vlabench": str(VLABENCH),
                "deploy_config": None if backend == "metis" else str(OPENWAM / "configs/deploy.yaml"),
                "client": str(METIS_CLIENT if backend == "metis" else BENCH / "single_eval.py"),
                "client_config": str(BENCH / "policy_config.yml"),
                "n_jobs": len(jobs), "n_episodes": sum(j["budget"] for j in jobs), "gpus": gpus,
                "servers_per_gpu": servers_per_gpu, "started": stamp(), "jobs": jobs}
    write_json(out / "campaign.json", manifest)
    n_slots = len(gpus) * servers_per_gpu
    servers = [Server(i, gpus[i % len(gpus)], port_base + i, ckpt_dir, ckpt_name, out, backend, metis)
               for i in range(n_slots)]
    stop = threading.Event()
    try:
        todo = pending(out, jobs)
        log(f"{len(jobs)} jobs, {len(todo)} pending, {n_slots} slots on GPUs {','.join(gpus)}")
        if todo:
            start_servers(servers, gpus)
        for pass_idx in range(1, max_passes + 1):
            todo = pending(out, jobs)
            if not todo:
                break
            log(f"pass {pass_idx}: {len(todo)} jobs")
            q: queue.Queue = queue.Queue()
            for job in todo:
                q.put(job)
            threads = [threading.Thread(target=slot_loop, args=(out, s, q, pass_idx, stop), daemon=True)
                       for s in servers]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            summarize(out)
    finally:
        stop.set()
        for s in servers:
            s.stop()
    left = pending(out, jobs)
    log(f"campaign finished: {len(jobs) - len(left)}/{len(jobs)} jobs accepted")
    summarize(out)
    return 0 if not left else 1


# ---------------------------------------------------------------------------------------------------------------
# summary


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def merge(out: Path, parts: list[Path]) -> None:
    """One campaign directory from campaigns of the same checkpoint split over hosts by ``--tracks`` / ``--tasks``."""
    manifests = [json.loads((p / "campaign.json").read_text()) for p in parts]
    manifest = {**manifests[0], "jobs": [j for m in manifests for j in m["jobs"]], "merged_from": [str(p) for p in parts]}
    (out / "status").mkdir(parents=True, exist_ok=True)
    write_json(out / "campaign.json", manifest)
    for part, m in zip(parts, manifests):
        for job in m["jobs"]:
            if status_path(part, job).exists():
                shutil.copy2(status_path(part, job), status_path(out, job))


def summarize(out: Path) -> dict:
    manifest = json.loads((out / "campaign.json").read_text())
    jobs = manifest["jobs"]
    tracks = {}
    present = {j["track"] for j in jobs}
    for track in [t for t in TRACKS if t in present] + sorted(present - set(TRACKS)):
        rows = []
        for job in (j for j in jobs if j["track"] == track):
            path = status_path(out, job)
            if path.exists():
                rows.append(json.loads(path.read_text()))
        per_task = {r["task"]: {**r["metrics"], "records": r["records"], "budget": r["budget"]} for r in rows}
        n_tasks = sum(1 for j in jobs if j["track"] == track)
        tracks[track] = {
            "tasks_done": len(rows), "tasks_total": n_tasks,
            "episodes": sum(r["records"] for r in rows), "dropped": sum(r["dropped"] for r in rows),
            "mean": {k: _mean([m[k] for m in per_task.values()]) for k in METRICS},
            "nan_tasks": {k: sorted(t for t, m in per_task.items()
                                    if m[k] is None or (isinstance(m[k], float) and math.isnan(m[k])))
                          for k in METRICS},
            "per_task": per_task,
        }
    complete = all(t["tasks_done"] == t["tasks_total"] for t in tracks.values())
    scored = [t for name, t in tracks.items() if name in TRACKS] or list(tracks.values())
    avg = {k: _mean([t["mean"][k] for t in scored]) if complete else None for k in METRICS}
    summary = {"checkpoint": manifest["checkpoint"], "updated": stamp(), "complete": complete,
               "avg": avg, "tracks": tracks, "paper_openwam_alpha": PAPER_ALPHA}
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(summary))
    return summary


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


def summary_markdown(s: dict) -> str:
    lines = [f"# VLABench closed-loop evaluation ({s['updated']})", "", f"checkpoint: `{s['checkpoint']}`", "",
             "| track | tasks | episodes (dropped) | SR | PS | IS | paper SR / PS / IS |", "|---|---|---|---|---|---|---|"]
    for track, t in s["tracks"].items():
        m = t["mean"]
        paper = " / ".join(f"{v:.1f}" for v in PAPER_ALPHA[track]) if track in PAPER_ALPHA else "–"
        lines.append(f"| {SHORT[track]} | {t['tasks_done']}/{t['tasks_total']} | {t['episodes']} ({t['dropped']}) | "
                     f"{_pct(m['success_rate'])} | {_pct(m['progress_score'])} | {_pct(m['intention_score'])} | "
                     f"{paper} |")
    a = s["avg"]
    paper = " / ".join(f"{v:.1f}" for v in PAPER_ALPHA["avg"])
    lines.append(f"| **Avg** | | | {_pct(a['success_rate'])} | {_pct(a['progress_score'])} | "
                 f"{_pct(a['intention_score'])} | {paper} |")
    shown = list(s["tracks"])
    lines += ["", "Per task SR (%):", "", "| task | " + " | ".join(SHORT[t] for t in shown) + " |",
              "|---|" + "---|" * len(shown)]
    tasks = sorted({task for t in s["tracks"].values() for task in t["per_task"]})
    for task in tasks:
        cells = []
        for track in shown:
            m = s["tracks"][track]["per_task"].get(task)
            cells.append("" if m is None else _pct(m["success_rate"]))
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary", "merge"])
    ap.add_argument("--parts", help="merge: comma-separated campaign directories")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt-dir", default=str(ALPHA_VLABENCH))
    ap.add_argument("--ckpt-name", default="checkpoint_step_6000.safetensors")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--servers-per-gpu", type=int, default=2)
    ap.add_argument("--port-base", type=int, default=8880)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--max-passes", type=int, default=4)
    ap.add_argument("--tracks", default=",".join(TRACKS))
    ap.add_argument("--tasks", default=None, help="comma-separated subset (default: every task of each track)")
    ap.add_argument("--backend", choices=["alpha", "metis"], default="alpha")
    ap.add_argument("--model-file", default=None, help="metis: bf16 state_dict from scripts/eval/export_bf16.py")
    ap.add_argument("--execute-steps", type=int, default=32)
    ap.add_argument("--policy", default="{}", help="metis: JSON sampler options")
    ap.add_argument("--policy-seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.output)
    if args.mode == "merge":
        merge(out, [Path(p) for p in args.parts.split(",")])
        args.mode = "summary"
    if args.mode == "summary":
        print(summary_markdown(summarize(out)))
        return
    metis = None
    if args.backend == "metis":
        metis = {"model_file": args.model_file, "execute_steps": args.execute_steps,
                 "policy": json.loads(args.policy), "policy_seed": args.policy_seed}
    raise SystemExit(run(out, Path(args.ckpt_dir), args.ckpt_name, args.gpus.split(","), args.servers_per_gpu,
                         args.port_base, args.n_episodes, args.max_passes, tuple(args.tracks.split(",")),
                         None if args.tasks is None else set(args.tasks.split(",")), args.backend, metis))


if __name__ == "__main__":
    main()
