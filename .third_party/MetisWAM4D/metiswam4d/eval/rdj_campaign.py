"""RoboDojo closed-loop campaign for MetisWAM4D: 42 base tasks / 54 task configs / 420 episodes (the local reduced
protocol: standalone tasks 10 episodes, the 12 Generalization tasks 5 standard + 5 random), evaluated **in rounds**:
every task config gets ``episodes_per_round`` more episodes before any config gets its next ones, so the running
numbers after every completed round are balanced over the whole benchmark.

Sub-commands (from the project root, ``/usr/bin/python3.10 -m metiswam4d.eval.rdj_campaign``):
    prepare   freeze ``campaign.json`` (variants, protocol, model file, options)
    run       one manager per host: one policy server per GPU (``rdj_policy.RDJServer``) and ``--clients`` official
              Isaac clients per GPU, each claiming the task config with the lowest pending round; a claimed config runs
              one round (``EVAL_NUM`` = the round's cumulative target) through the official ``eval_client`` with the
              resume manifest rebuilt from ``_result.json``, so the official per-layout seeds continue in order
    summary   ``summary.json`` / ``summary.md``: five-dimension macro averages over the completed rounds (each config
              cut to the same round) and over every finished episode

Layout: ``<out>/results/RoboDojo/<variant>/MetisWAM4D/arx_x5/<seed>_<info>/<variant>/_result.json`` (official),
``<out>/claims/``, ``<out>/logs/``, ``<out>/gpuN/observations`` (first policy input of every episode).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
JANUS = Path("/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824")
ROBODOJO = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo")
POLICY_PYTHON = "/usr/bin/python3.10"
CLIENT_PYTHON = "/usr/local/robodojo-python-runtimes/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
# SAPIEN (policy-side robot geometry) Vulkan ICD.  The GLX ICD works on the BCC pods (t2, dev); the IDC machine d1
# needs SAPIEN's EGL ICD (10_nvidia.json), selected per host through METIS_SAPIEN_ICD.
SAPIEN_ICD = os.environ.get("METIS_SAPIEN_ICD",
                            "/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json")
POLICY_NAME = "MetisWAM4D"
CONFIG_NAME = "arx_x5"
PROXY = "http://oversea-squid1.jp.txyun:11080"
NO_PROXY = "localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com"
DIM_ORDER = ("Generalization", "Precision", "Long-Horizon", "Memory", "Open")
# Local 420-episode protocol, same simulator and scoring (the evaluation records of JanusTrack4d_260824).
BASELINES = {
    "JanusAct4D-RDJ-ipft 40k": {"Overall SR": 10.75, "Score": 16.82, "Generalization": 11.67, "Gen-Std": 23.33,
                                "Gen-Random": 0.0, "Precision": 18.75, "Long-Horizon": 15.00, "Memory": 8.33, "Open": 0.0},
    "OpenWAM-alpha 60k (local)": {"Overall SR": 9.58, "Score": 15.40, "Generalization": 10.00, "Gen-Std": 13.33,
                                  "Gen-Random": 6.67, "Precision": 7.50, "Long-Horizon": 22.50, "Memory": 6.67, "Open": 1.25},
    "OpenWAM-alpha (leaderboard)": {"Overall SR": 11.92, "Score": 17.18, "Generalization": 14.83, "Gen-Std": 25.56,
                                    "Gen-Random": 4.11, "Precision": 9.25, "Long-Horizon": 25.33, "Memory": 9.11, "Open": 1.08},
}


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def load_reference(name: str):
    path = ROBODOJO / "experiments/fastwam_robodojo_10ep" / (name + ".py")
    spec = importlib.util.spec_from_file_location("robodojo_reference_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------------------------

def layout_count(task: str, env_seed: int) -> int:
    """Official scene layouts of a task config under ``Assets/Eval_Layout/RoboDojo/arx_x5/<env_seed>/``."""
    return len(glob.glob(str(ROBODOJO / "Assets/Eval_Layout/RoboDojo" / CONFIG_NAME / str(env_seed) / f"{task}_[0-9]*.json")))


def prepare(out: Path, config: str, model_file: str, source_checkpoint: str, episodes_per_round: int,
            execute_steps: int, clients_per_gpu: int, policy: dict, info: str, only: list[str] | None,
            episodes: int | None, video_episodes: int, policy_seed: int, policy_variants: dict | None = None,
            layout_offset: int = 0, record_dir: str | None = None, env_seed: int = 0,
            obs_every_step: bool = False, exclude_layouts: dict | None = None) -> dict:
    """``policy_variants`` = {label: policy options}: every task config is evaluated under every label (names
    ``<task>@<label>``, same layouts -> paired).  ``layout_offset`` skips the first layouts (the formal protocol uses
    0-9; a development set uses 10+).  ``exclude_layouts`` = {task config: [layout ids]} skips further layouts of a
    config (e.g. layouts whose self-play rollouts are in the training set).  ``record_dir`` makes the clients write
    the trajectories (self-play data).  ``env_seed`` selects the official layout set (the protocol uses 0; sets 1 and
    2 are disjoint layouts).  The episode count of a config never exceeds its available layouts (skipped excluded)."""
    protocol = load_reference("eval_protocol").resolve_protocol(ROBODOJO, {"standalone_episodes": 10,
                                                                          "paired_half_episodes": 5})
    variants = [{**v, "task": v["name"], "policy": {}} for v in protocol["variants"]]
    if only:
        variants = [v for v in variants if v["name"] in set(only)]
        if len(variants) != len(set(only)):
            raise ValueError(f"unknown task config in {only}")
    if episodes is not None:
        variants = [{**v, "episodes": int(episodes)} for v in variants]
    exclude_layouts = exclude_layouts or {}
    variants = [{**v, "exclude_layouts": sorted(int(i) for i in exclude_layouts.get(v["task"], []))} for v in variants]
    variants = [{**v, "episodes": max(0, min(int(v["episodes"]), layout_count(v["task"], env_seed)
                                             - len(set(range(layout_offset)) | set(v["exclude_layouts"]))))}
                for v in variants]
    if policy_variants:
        variants = [{**v, "name": f"{v['name']}@{label}", "policy": dict(options), "label": label}
                    for v in variants for label, options in policy_variants.items()]
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": str(Path(config).resolve()),
        "model_file": str(model_file), "source_checkpoint": str(source_checkpoint), "robodojo_root": str(ROBODOJO),
        "policy_name": POLICY_NAME, "config_name": CONFIG_NAME, "additional_info": info, "env_seed": int(env_seed),
        "protocol": {"execute_steps": execute_steps, "rounds": 20, "stop_after": "action (round 10)",
                     "policy": policy, "policy_seed": policy_seed,
                     "episode_seed": "sha256(metiswam4d-rdj:policy_seed:task_config_without_@label:layout_id); re-plan k: sha256(seed:replan:k)",
                     "observation": ("one observation after every action (official deploy cadence)" if obs_every_step or record_dir
                                     else "one observation per executed chunk, render_sync kit flags"),
                     "obs_every_step": bool(obs_every_step or record_dir),
                     "geometry": "SAPIEN robot-only depth (mm) + mask from live joint states (training renderer)",
                     "rgb": "640x480 -> 320x240 INTER_AREA -> JPEG q95 (cv2 on RGB) -> dataset decode",
                     "video_episodes": video_episodes, "episodes_per_round": episodes_per_round,
                     "layout_offset": int(layout_offset)},
        "policy_variants": policy_variants or None, "record_dir": record_dir,
        "clients_per_gpu": clients_per_gpu, "dimensions": protocol["dimensions"], "variants": variants,
    }
    write_json(out / "campaign.json", manifest)
    return manifest


# ----------------------------------------------------------------------------------------------------------------
# results on disk
# ----------------------------------------------------------------------------------------------------------------

def task_of(variant) -> str:
    name = variant["name"] if isinstance(variant, dict) else variant
    return variant.get("task", name.split("@")[0]) if isinstance(variant, dict) else name.split("@")[0]


def save_dir(out: Path, campaign: dict, variant) -> Path:
    """The official result directory of a task config (``build_result_dir`` layout, run id = variant name)."""
    name = variant["name"] if isinstance(variant, dict) else variant
    return (out / "results/RoboDojo" / task_of(variant) / campaign["policy_name"] / campaign["config_name"]
            / f"{campaign['env_seed']}_{campaign['additional_info']}" / name)


def result_file(out: Path, variant) -> Path | None:
    name = variant["name"] if isinstance(variant, dict) else variant
    paths = glob.glob(str(out / "results/RoboDojo" / task_of(variant) / "**" / name / "_result.json"), recursive=True)
    if not paths:
        return None
    if len(paths) > 1:
        raise RuntimeError(f"several result files for {name}: {paths}")
    return Path(paths[0])


def read_details(out: Path, variant) -> list[dict]:
    """Finished episodes of a task config in completion order (``[{layout_id, success, score}]``)."""
    path = result_file(out, variant)
    if path is None:
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    details = data.get("details") or {}
    return [details[k] for k in sorted(details, key=int)]


def round_target(variant: dict, rnd: int, per_round: int) -> int:
    return min(int(rnd) * per_round, int(variant["episodes"]))


def next_round(variant: dict, done: int, per_round: int) -> int | None:
    """1-based index of the round that would add episodes to this config; None when complete."""
    if done >= variant["episodes"]:
        return None
    return done // per_round + 1


def synthesize_resume_manifest(result_path: Path | None, campaign: dict, variant, layout_offset: int = 0,
                               directory: Path | None = None) -> Path | None:
    """The official client deletes its resume manifest on normal completion; rebuild it from ``_result.json`` so
    the next round continues with the next layouts (existing manifests — a crashed client — are kept).
    ``layout_offset`` marks layouts ``0 .. offset-1`` and the variant's ``exclude_layouts`` as abandoned (skipped by
    the seed manager)."""
    directory = result_path.parent if result_path is not None else directory
    manifest = directory.parent / f"_resume_{directory.name}.json"
    if manifest.exists():
        return manifest
    details = (json.loads(result_path.read_text()).get("details") or {}) if result_path is not None else {}
    abandoned = sorted(set(range(int(layout_offset)))
                       | set(variant.get("exclude_layouts", []) if isinstance(variant, dict) else []))
    if not details and not abandoned:
        return None
    ordered = {str(k): details[k] for k in sorted(details, key=int)}
    successes = sum(1 for v in ordered.values() if v.get("success"))
    payload = {
        "run_id": directory.name, "save_dir": str(directory), "task_name": task_of(variant),
        "policy_name": campaign["policy_name"], "config_name": campaign["config_name"], "eval_seed": campaign["env_seed"],
        "additional_info": campaign["additional_info"], "success_nums": successes,
        "fail_nums": len(ordered) - successes, "unstable_nums": 0,
        "total_score": float(sum(float(v.get("score", 0.0)) for v in ordered.values())),
        "completed_layout_ids": sorted(int(v["layout_id"]) for v in ordered.values()),
        "abandoned_layout_ids": abandoned, "details": ordered, "restart_count": 0,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(manifest)
    return manifest


# ----------------------------------------------------------------------------------------------------------------
# claims (exclusive files on the shared disk; one task config runs in one client at a time)
# ----------------------------------------------------------------------------------------------------------------

def claim_path(out: Path, variant: str) -> Path:
    return out / "claims" / f"{variant}.claim"


def try_claim(out: Path, variant: str, tag: str) -> bool:
    path = claim_path(out, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"tag": tag, "host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}))
    return True


def release_claim(out: Path, variant: str) -> None:
    claim_path(out, variant).unlink(missing_ok=True)


def release_host_claims(out: Path, host: str) -> int:
    """Drop the claims of managers that no longer run on this host (their pid is dead); claims of other hosts and
    of a second live manager on this host are kept."""
    released = 0
    for path in glob.glob(str(out / "claims" / "*.claim")):
        try:
            claim = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if claim.get("host") != host:
            continue
        try:
            os.kill(int(claim.get("pid", 0)), 0)
            alive = True
        except (OSError, ValueError):
            alive = False
        if not alive:
            Path(path).unlink(missing_ok=True)
            released += 1
    return released


# ----------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------

def policy_env(gpu: int) -> dict:
    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=os.pathsep.join([str(PROJECT), str(PROJECT / "third_party/msgpack_numpy"), str(JANUS),
                                    str(ROBODOJO), str(ROBODOJO / "XPolicyLab")]),
        VK_ICD_FILENAMES=SAPIEN_ICD,
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def client_env(gpu: int, campaign: dict, out: Path, variant: dict, target: int) -> dict:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), ROBODOJO_ROOT=str(ROBODOJO), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
        EVAL_ENV_TYPE="sim", ROBODOJO_RECORD_VIDEO="1" if campaign["protocol"]["video_episodes"] > 0 else "0",
        METIS_VIDEO_EPISODES=str(campaign["protocol"]["video_episodes"]),
        ROBODOJO_EVAL_ROOT=str(out / "results"), ROBODOJO_RUN_ID=variant["name"], EVAL_NUM=str(target),
        METIS_VARIANT=variant["name"], METIS_POLICY=json.dumps(variant.get("policy") or {}),
        METIS_RECORD_DIR=str(campaign.get("record_dir") or ""),
        METIS_OBS_EVERY_STEP="1" if campaign["protocol"].get("obs_every_step") else "0",
        # self-play collection runs past the official per-task episode caps (_task.yml eval_nums 25 / 50)
        METIS_UNCAP_EVAL_NUM="1" if campaign.get("record_dir") else "0",
        http_proxy=PROXY, https_proxy=PROXY, HTTP_PROXY=PROXY, HTTPS_PROXY=PROXY, no_proxy=NO_PROXY, NO_PROXY=NO_PROXY,
        PYTHONPATH=os.pathsep.join([str(PROJECT), str(ROBODOJO / "scripts/runtime_sitecustomize"), str(ROBODOJO),
                                    str(ROBODOJO / "XPolicyLab"), env.get("PYTHONPATH", "")]))
    return env


def client_command(port: int, campaign: dict, out: Path, variant: dict) -> list[str]:
    kit = ("--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
           f"--/log/file={out / 'logs' / (variant['name'] + '_kit.log')} --/app/settings/persistent=false")
    # render synchronisation before each observation; METIS_KIT_RENDER_SYNC=0 keeps the official client's asynchronous
    # rendering (frames may lag the physics state by one step)
    if os.environ.get("METIS_KIT_RENDER_SYNC", "1") != "0":
        kit += (" --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true"
                " --/app/hydraEngine/waitIdle=true")
    # host-specific Kit settings, e.g. "--/rtx/verifyDriverVersion/enabled=false" on nodes whose driver is older
    # than the Omniverse RTX minimum (d1, 535.54.03)
    if os.environ.get("METIS_EXTRA_KIT_ARGS"):
        kit += " " + os.environ["METIS_EXTRA_KIT_ARGS"]
    return [CLIENT_PYTHON, "-m", "metiswam4d.eval.rdj_client",
            "--task_name", task_of(variant), "--env_cfg_type", CONFIG_NAME, "--num_envs", "1", "--enable_cameras",
            "--headless", "--device_id", "0", "--policy_name", POLICY_NAME, "--port", str(port), "--host", "127.0.0.1",
            "--protocol", "ws", "--additional_info", campaign["additional_info"], "--seed", str(campaign["env_seed"]),
            "--kit_args", kit]


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def gpu_snapshot() -> list[dict]:
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in text.strip().splitlines():
        idx, util, used, total = [p.strip() for p in line.split(",")]
        rows.append({"gpu": int(idx), "util": int(util), "mem_used": int(used), "mem_total": int(total)})
    return rows


class Manager:
    def __init__(self, out: Path, gpus: list[int], clients: int, base_port: int, stall_seconds: int,
                 keepalive_pid_file: str | None, min_free_mb: int):
        self.out = out
        self.campaign = json.loads((out / "campaign.json").read_text())
        self.variants = {v["name"]: v for v in self.campaign["variants"]}
        self.per_round = int(self.campaign["protocol"]["episodes_per_round"])
        self.gpus, self.clients, self.base_port = gpus, clients, base_port
        self.stall_seconds, self.min_free_mb = stall_seconds, min_free_mb
        self.keepalive_pid_file = keepalive_pid_file
        self.host = socket.gethostname()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.children: set[subprocess.Popen] = set()
        self.servers: dict[int, subprocess.Popen] = {}
        self.server_locks = {gpu: threading.Lock() for gpu in gpus}
        self.active: dict[str, dict] = {}       # variant -> {gpu, slot, target, started}
        (out / "logs").mkdir(parents=True, exist_ok=True)
        self.log_file = open(out / "logs" / f"manager_{self.host}.log", "a")

    # -- logging -------------------------------------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    # -- processes -----------------------------------------------------------------------------------------------
    def launch(self, command: list[str], env: dict, log_path: Path) -> subprocess.Popen:
        log = open(log_path, "a")
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        process._log_handle = log  # closed with the process
        with self.lock:
            self.children.add(process)
        return process

    def wait_port_free(self, port: int) -> None:
        for _ in range(90):
            try:
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
                return
            except OSError:
                if self.stop.wait(5):
                    return
        raise RuntimeError(f"port {port} stays busy")

    def start_server(self, gpu: int) -> subprocess.Popen:
        port = self.base_port + gpu
        self.wait_port_free(port)
        (self.out / f"gpu{gpu}").mkdir(exist_ok=True)
        server = self.launch([POLICY_PYTHON, "-m", "metiswam4d.eval.rdj_policy", "--campaign",
                              str(self.out / "campaign.json"), "--output", str(self.out / f"gpu{gpu}"), "--port", str(port)],
                             policy_env(gpu), self.out / "logs" / f"policy_gpu{gpu}_{self.host}.log")
        from websockets.sync.client import connect
        deadline = time.monotonic() + 1800
        while not self.stop.is_set():
            if server.poll() is not None:
                raise RuntimeError(f"GPU {gpu} policy server exited with {server.returncode}")
            try:
                with connect(f"ws://127.0.0.1:{port}", open_timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"GPU {gpu} policy server not ready after 30 min")
                self.stop.wait(3)
        self.log(f"policy server gpu{gpu} ready on port {port} (pid {server.pid})")
        return server

    def ensure_server(self, gpu: int) -> subprocess.Popen:
        with self.server_locks[gpu]:
            server = self.servers.get(gpu)
            if server is not None and server.poll() is None:
                return server
            if server is not None:
                self.log(f"policy server gpu{gpu} died with {server.returncode}; restarting")
                with self.lock:
                    self.children.discard(server)
            server = self.start_server(gpu)
            self.servers[gpu] = server
            return server

    # -- work selection --------------------------------------------------------------------------------------------
    def progress(self) -> dict[str, int]:
        return {name: len(read_details(self.out, name)) for name in self.variants}

    def claim_next(self, tag: str) -> tuple[dict, int] | None:
        """The unclaimed task config with the lowest pending round (ties: the most remaining simulation steps)."""
        done = self.progress()
        candidates = []
        for name, variant in self.variants.items():
            rnd = next_round(variant, done[name], self.per_round)
            if rnd is None or claim_path(self.out, name).exists():
                continue
            remaining = (variant["episodes"] - done[name]) * variant["step_limit"]
            candidates.append((rnd, -remaining, name))
        for rnd, _, name in sorted(candidates):
            if try_claim(self.out, name, tag):
                return self.variants[name], round_target(self.variants[name], rnd, self.per_round)
        return None

    def pending_anywhere(self) -> bool:
        done = self.progress()
        return any(next_round(v, done[n], self.per_round) is not None for n, v in self.variants.items())

    # -- one round of one task config -------------------------------------------------------------------------------
    def run_chunk(self, gpu: int, slot: int, variant: dict, target: int) -> bool:
        name = variant["name"]
        port = self.base_port + gpu
        log_path = self.out / "logs" / f"{name}.log"
        disable_graphs = False
        for attempt in range(4):
            if self.stop.is_set():
                return False
            self.ensure_server(gpu)
            path = result_file(self.out, variant)
            offset = int(self.campaign["protocol"].get("layout_offset", 0))
            if path is not None or offset > 0 or variant.get("exclude_layouts"):
                synthesize_resume_manifest(path, self.campaign, variant, offset, save_dir(self.out, self.campaign, variant))
            before = len(read_details(self.out, variant))
            if before >= target:
                return True
            env = client_env(gpu, self.campaign, self.out, variant, target)
            if disable_graphs:
                env["METIS_DISABLE_CUROBO_GRAPHS"] = "1"
            with open(log_path, "a") as handle:
                handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.host} gpu{gpu} slot{slot} "
                             f"episodes {before} -> {target} attempt {attempt}\n")
            offset = os.path.getsize(log_path)
            client = self.launch(client_command(port, self.campaign, self.out, variant), env, log_path)
            self.log(f"gpu{gpu}/s{slot} {name}: episodes {before} -> {target} (pid {client.pid}, attempt {attempt})")
            episodes, advanced, stalled = before, time.monotonic(), False
            while client.poll() is None and not self.stop.wait(15):
                current = len(read_details(self.out, name))
                if current != episodes:
                    episodes, advanced = current, time.monotonic()
                elif time.monotonic() - advanced > self.stall_seconds:
                    stalled = True
                    self.log(f"gpu{gpu}/s{slot} {name}: no episode in {self.stall_seconds}s, terminating client")
                    stop_process(client)
                    break
                with self.lock:
                    server = self.servers.get(gpu)
                if server is not None and server.poll() is not None:
                    self.log(f"gpu{gpu}/s{slot} {name}: policy server gone, terminating client")
                    stop_process(client)
                    break
            if self.stop.is_set():
                stop_process(client)
                return False
            with self.lock:
                self.children.discard(client)
            client._log_handle.close()
            done = len(read_details(self.out, name))
            if done >= target and client.returncode == 0:
                return True
            with open(log_path) as handle:
                handle.seek(offset)
                tail = handle.read()
            capture_failed = "cudaErrorStreamCaptureUnsupported" in tail
            self.log(f"gpu{gpu}/s{slot} {name}: client exit {client.returncode}, {done}/{target} episodes"
                     f"{', stalled' if stalled else ''}{', graph capture failed' if capture_failed else ''}")
            if capture_failed:
                disable_graphs = True
            self.stop.wait(20)
        return len(read_details(self.out, name)) >= target

    # -- slot loop -----------------------------------------------------------------------------------------------
    def slot_loop(self, gpu: int, slot: int) -> None:
        tag = f"{self.host}:gpu{gpu}:s{slot}"
        idle_since = None
        while not self.stop.is_set():
            if not self.memory_ok(gpu):
                self.log(f"gpu{gpu}/s{slot}: less than {self.min_free_mb} MB free, waiting")
                self.stop.wait(60)
                continue
            claimed = self.claim_next(tag)
            if claimed is None:
                if not self.pending_anywhere():
                    self.log(f"gpu{gpu}/s{slot}: campaign complete")
                    return
                if idle_since is None:
                    idle_since = time.monotonic()
                self.stop.wait(60)
                continue
            idle_since = None
            variant, target = claimed
            with self.lock:
                self.active[variant["name"]] = {"gpu": gpu, "slot": slot, "target": target, "started": time.time()}
            try:
                ok = self.run_chunk(gpu, slot, variant, target)
                self.log(f"gpu{gpu}/s{slot} {variant['name']}: round {'done' if ok else 'incomplete'} "
                         f"({len(read_details(self.out, variant['name']))}/{variant['episodes']})")
            except Exception:
                self.log(f"gpu{gpu}/s{slot} {variant['name']}: {traceback.format_exc()}")
                self.stop.wait(30)
            finally:
                with self.lock:
                    self.active.pop(variant["name"], None)
                release_claim(self.out, variant["name"])

    def memory_ok(self, gpu: int) -> bool:
        for row in gpu_snapshot():
            if row["gpu"] == gpu:
                return row["mem_total"] - row["mem_used"] >= self.min_free_mb
        return True

    # -- watchdog ------------------------------------------------------------------------------------------------
    def keepalive_alive(self) -> bool:
        """The GPU keep-alive (``wangrunqi_nvml_busy.py``) must stay up through the campaign: it fills the
        utilisation gaps the simulators leave (the queue monitor reclaims idle GPUs)."""
        if self.keepalive_pid_file and Path(self.keepalive_pid_file).exists():
            try:
                pid = int(Path(self.keepalive_pid_file).read_text().split("pid=")[1].split()[0])
                os.kill(pid, 0)
                return True
            except (OSError, IndexError, ValueError):
                pass
        probe = subprocess.run(["pgrep", "-f", "wangrunqi_nvml_busy[.]py"], capture_output=True, text=True)
        return bool(probe.stdout.strip())

    def restart_keepalive(self) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"rdjeval_{self.host.split('.')[0]}_{stamp}"
        logs = Path("/ytech_milm_intern/danglingwei/logs")
        subprocess.Popen([POLICY_PYTHON, "wangrunqi_nvml_busy.py", "--gpus", "all", "--size", "2000",
                          "--pid-file", str(logs / f"{name}.pid"), "--log-file", str(logs / f"{name}.log")],
                         cwd="/ytech_milm_intern/danglingwei", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        self.keepalive_pid_file = str(logs / f"{name}.pid")
        self.log(f"keep-alive restarted ({self.keepalive_pid_file})")

    def watchdog(self) -> None:
        last_summary = 0.0
        while not self.stop.wait(60):
            rows = gpu_snapshot()
            with open(self.out / "logs" / f"gpu_monitor_{self.host}.jsonl", "a") as handle:
                handle.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpus": rows,
                                         "active": len(self.active)}) + "\n")
            low = [r for r in rows if r["gpu"] in self.gpus and r["util"] < 20]
            if low:
                self.log(f"low GPU utilisation: {[(r['gpu'], r['util']) for r in low]}")
            alive = self.keepalive_alive()
            if not alive:
                self.log("keep-alive process is not running")
                self.restart_keepalive()
            for gpu in self.gpus:
                with self.lock:
                    server = self.servers.get(gpu)
                if server is not None and server.poll() is not None and any(
                        a["gpu"] == gpu for a in self.active.values()):
                    self.log(f"policy server gpu{gpu} exited with {server.returncode} (a slot will restart it)")
            with self.lock:
                write_json(self.out / f"status_{self.host.split('.')[0]}_{os.getpid()}.json", {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "active": self.active,
                    "servers": {g: (s.pid if s is not None and s.poll() is None else None) for g, s in self.servers.items()},
                    "keepalive_alive": alive, "gpus": rows})
            if time.time() - last_summary > 300:
                try:
                    summary(self.out)
                except Exception:
                    self.log("summary failed: " + traceback.format_exc().splitlines()[-1])
                last_summary = time.time()

    # -- main ----------------------------------------------------------------------------------------------------
    def run(self) -> int:
        released = release_host_claims(self.out, self.host)
        if released:
            self.log(f"released {released} stale claims of {self.host}")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.stop.set())
        self.log(f"manager start: gpus {self.gpus}, {self.clients} clients/GPU, model {self.campaign['model_file']}")
        threads = [threading.Thread(target=self.watchdog, daemon=True)]
        for gpu in self.gpus:
            for slot in range(self.clients):
                threads.append(threading.Thread(target=self.slot_loop, args=(gpu, slot), daemon=True))
        for thread in threads:
            thread.start()
        try:
            for thread in threads[1:]:
                while thread.is_alive():
                    thread.join(5)
        finally:
            self.stop.set()
            with self.lock:
                children = list(self.children)
            for child in children:
                stop_process(child)
            try:
                summary(self.out)
            except Exception:
                pass
        complete = not self.pending_anywhere()
        self.log("manager exit: " + ("campaign complete" if complete else "interrupted"))
        return 0 if complete else 1


# ----------------------------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------------------------

def aggregate(campaign: dict, details: dict[str, list[dict]], cut: dict[str, int] | None) -> dict:
    """Five-dimension macro averages.  ``cut`` = episodes to keep per task config (None = all finished)."""
    variants = campaign["variants"]
    per_variant, per_task = {}, {}
    for v in variants:
        rows = details.get(v["name"], [])
        if cut is not None:
            rows = rows[:cut[v["name"]]]
        n = len(rows)
        succ = sum(1 for r in rows if r.get("success"))
        score = float(sum(float(r.get("score", 0.0)) for r in rows))
        per_variant[v["name"]] = {"n": n, "success": succ, "score_sum": score, "split": v["split"],
                                  "dimension": v["dimension"], "base_task": v["base_task"]}
        t = per_task.setdefault(v["base_task"], {"n": 0, "success": 0, "score_sum": 0.0, "dimension": v["dimension"]})
        t["n"] += n
        t["success"] += succ
        t["score_sum"] += score
    dims = {}
    for dim, tasks in campaign["dimensions"].items():
        rows = [per_task[t] for t in tasks if t in per_task and per_task[t]["n"] > 0]
        dims[dim] = {"tasks": len(rows), "sr": 100 * sum(r["success"] / r["n"] for r in rows) / len(rows) if rows else None,
                     "score": 100 * sum(r["score_sum"] / r["n"] for r in rows) / len(rows) if rows else None}
    for split, label in (("standard", "Gen-Std"), ("random", "Gen-Random")):
        rows = [r for r in per_variant.values() if r["dimension"] == "Generalization" and r["split"] == split and r["n"] > 0]
        dims[label] = {"tasks": len(rows), "sr": 100 * sum(r["success"] / r["n"] for r in rows) / len(rows) if rows else None,
                       "score": None}
    scored = [dims[d] for d in DIM_ORDER if d in dims and dims[d]["sr"] is not None]
    overall = {"sr": sum(d["sr"] for d in scored) / len(scored) if scored else None,
               "score": sum(d["score"] for d in scored) / len(scored) if scored else None,
               "dimensions_scored": len(scored)}
    total_n = sum(r["n"] for r in per_variant.values())
    total_s = sum(r["success"] for r in per_variant.values())
    return {"overall": overall, "dimensions": dims, "tasks": per_task, "variants": per_variant,
            "episodes": total_n, "successes": total_s, "micro_sr": 100 * total_s / total_n if total_n else None}


def complete_rounds(campaign: dict, details: dict[str, list[dict]]) -> int:
    per_round = int(campaign["protocol"]["episodes_per_round"])
    max_rounds = max((v["episodes"] + per_round - 1) // per_round for v in campaign["variants"])
    done = 0
    for rnd in range(1, max_rounds + 1):
        if all(len(details.get(v["name"], [])) >= round_target(v, rnd, per_round) for v in campaign["variants"]):
            done = rnd
        else:
            break
    return done


def fmt(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def summary_variants(out: Path, campaign: dict, details: dict[str, list[dict]]) -> dict:
    """Development campaign: the same task configs under several policy-option labels, paired by layout."""
    labels = list(campaign["policy_variants"])
    tasks = sorted({v["task"] for v in campaign["variants"]}, key=lambda t: (next(v["dimension"] for v in campaign["variants"] if v["task"] == t), t))
    table, totals = {}, {label: [0, 0, 0.0] for label in labels}
    paired = {label: [0, 0] for label in labels}   # over layouts finished under every label
    for task in tasks:
        row = {}
        per_label = {label: {int(d["layout_id"]): d for d in details.get(f"{task}@{label}", [])} for label in labels}
        common = set.intersection(*(set(per_label[label]) for label in labels)) if labels else set()
        for label in labels:
            rows = list(per_label[label].values())
            n, s = len(rows), sum(1 for r in rows if r.get("success"))
            row[label] = (s, n, float(sum(float(r.get("score", 0)) for r in rows)))
            totals[label][0] += s
            totals[label][1] += n
            totals[label][2] += row[label][2]
            paired[label][0] += sum(1 for l in common if per_label[label][l].get("success"))
            paired[label][1] += len(common)
        table[task] = row
    lines = [f"# RoboDojo development campaign — {Path(campaign['source_checkpoint']).name}", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}.  Layouts from {campaign['protocol'].get('layout_offset', 0)}; "
             f"policy variants: {json.dumps(campaign['policy_variants'])}", "",
             "| Task | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for task, row in table.items():
        lines.append(f"| {task} | " + " | ".join(f"{row[l][0]}/{row[l][1]}" for l in labels) + " |")
    lines.append("| **total** | " + " | ".join(f"**{totals[l][0]}/{totals[l][1]}** (score {100 * totals[l][2] / max(totals[l][1], 1):.1f})"
                                               for l in labels) + " |")
    lines.append("| paired (layouts finished under every label) | " + " | ".join(f"{paired[l][0]}/{paired[l][1]}" for l in labels) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    result = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "labels": labels, "tasks": table, "totals": totals,
              "paired": paired, "complete_rounds": complete_rounds(campaign, details),
              "all": {"episodes": sum(t[1] for t in totals.values())}}
    write_json(out / "summary.json", result)
    return result


def summary(out: Path) -> dict:
    campaign = json.loads((out / "campaign.json").read_text())
    details = {v["name"]: read_details(out, v) for v in campaign["variants"]}
    if campaign.get("policy_variants"):
        return summary_variants(out, campaign, details)
    per_round = int(campaign["protocol"]["episodes_per_round"])
    rounds = complete_rounds(campaign, details)
    cut = {v["name"]: round_target(v, rounds, per_round) for v in campaign["variants"]} if rounds else None
    balanced = aggregate(campaign, details, cut) if rounds else None
    everything = aggregate(campaign, details, None)
    total = sum(v["episodes"] for v in campaign["variants"])
    result = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "model_file": campaign["model_file"],
              "source_checkpoint": campaign["source_checkpoint"], "complete_rounds": rounds,
              "episodes_per_round": per_round, "target_episodes": total, "balanced": balanced, "all": everything}
    write_json(out / "summary.json", result)

    lines = [f"# RoboDojo closed loop — {Path(campaign['source_checkpoint']).name}", "",
             f"Updated {result['updated']}.  Complete rounds: **{rounds}** (every task config cut to "
             f"{'its round-' + str(rounds) + ' target' if rounds else 'nothing yet'}); finished episodes "
             f"{everything['episodes']}/{total}, successes {everything['successes']}.", ""]
    head = "| | MetisWAM4D (complete rounds) | MetisWAM4D (all finished) | " + " | ".join(BASELINES) + " |"
    lines += [head, "|---|---:|---:|" + "---:|" * len(BASELINES)]

    def cell(agg, key):
        if agg is None:
            return "—"
        if key == "Overall SR":
            return fmt(agg["overall"]["sr"])
        if key == "Score":
            return fmt(agg["overall"]["score"])
        return fmt(agg["dimensions"].get(key, {}).get("sr"))

    for key in ("Overall SR", "Score", "Generalization", "Gen-Std", "Gen-Random", "Precision", "Long-Horizon", "Memory", "Open"):
        lines.append(f"| {key} | {cell(balanced, key)} | {cell(everything, key)} | "
                     + " | ".join(fmt(b.get(key)) for b in BASELINES.values()) + " |")
    lines += ["", f"Micro SR over all finished episodes: {everything['successes']}/{everything['episodes']} = "
              f"{fmt(everything['micro_sr'])}%.", "", "## Task configs", "",
              "| Task config | Dimension | Done / target | Successes | Score |", "|---|---|---:|---:|---:|"]
    for v in campaign["variants"]:
        row = everything["variants"][v["name"]]
        lines.append(f"| {v['name']} | {v['dimension']} | {row['n']} / {v['episodes']} | {row['success']} | "
                     f"{fmt(100 * row['score_sum'] / row['n'] if row['n'] else None, 1)} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return result


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--episodes-per-round", type=int, default=2)
    p.add_argument("--execute-steps", type=int, default=32)
    p.add_argument("--clients-per-gpu", type=int, default=2)
    p.add_argument("--policy", default="{}", help="JSON: rounds / cfg / samples")
    p.add_argument("--info", default="metis")
    p.add_argument("--only", action="append")
    p.add_argument("--episodes", type=int, help="override the episode count of every task config (smoke)")
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--policy-seed", type=int, default=42)
    p.add_argument("--policy-variants", help="JSON {label: policy options}: paired development campaign")
    p.add_argument("--layout-offset", type=int, default=0, help="skip the first N official layouts (protocol = 0-9)")
    p.add_argument("--record", help="directory for the recorded trajectories (self-play data)")
    p.add_argument("--env-seed", type=int, default=0, help="official layout set (protocol 0; 1 / 2 are disjoint)")
    p.add_argument("--obs-every-step", action="store_true",
                   help="take an observation after every action (official deploy cadence) instead of once per chunk")
    p.add_argument("--exclude-layouts", help="JSON file {task config: [layout ids]} of further layouts to skip")
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--gpus", required=True)
    r.add_argument("--clients", type=int)
    r.add_argument("--base-port", type=int, default=33380)
    r.add_argument("--stall-seconds", type=int, default=3600)
    r.add_argument("--keepalive-pid-file")
    r.add_argument("--min-free-mb", type=int, default=12000)
    s = sub.add_parser("summary")
    s.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if args.command == "prepare":
        out.mkdir(parents=True, exist_ok=True)
        manifest = prepare(out, args.config, args.model_file, args.source_checkpoint, args.episodes_per_round,
                           args.execute_steps, args.clients_per_gpu, json.loads(args.policy), args.info, args.only,
                           args.episodes, args.video_episodes, args.policy_seed,
                           json.loads(args.policy_variants) if args.policy_variants else None, args.layout_offset,
                           args.record, args.env_seed, args.obs_every_step,
                           json.loads(Path(args.exclude_layouts).read_text()) if args.exclude_layouts else None)
        print(json.dumps({"variants": len(manifest["variants"]),
                          "episodes": sum(v["episodes"] for v in manifest["variants"]), "output": str(out)}))
    elif args.command == "run":
        campaign = json.loads((out / "campaign.json").read_text())
        manager = Manager(out, [int(g) for g in args.gpus.split(",")], args.clients or campaign["clients_per_gpu"],
                          args.base_port, args.stall_seconds, args.keepalive_pid_file, args.min_free_mb)
        sys.exit(manager.run())
    else:
        result = summary(out)
        print((out / "summary.md").read_text())
        print(json.dumps({"complete_rounds": result["complete_rounds"], "episodes": result["all"]["episodes"]}))


if __name__ == "__main__":
    main()
