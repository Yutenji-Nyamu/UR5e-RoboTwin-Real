#!/usr/bin/env python
"""Closed-loop RoboTwin probes: the same K/V interventions as the open-loop pass, but scored by success rate.

Re-uses the previous project's evaluation stack (``janusact4d_rt2imperfect_v1.evaluation.worker.sim_loop``:
one resident model per GPU + ``sims_per_gpu`` simulator processes, protocol identical to ``simeval_19045``:
10 sync denoising steps, execute 24 of 32 actions (8 for the dense tasks), frozen environment seeds and
instructions) and only swaps the policy for :class:`ProbePolicy`.

    source scripts/probe/env.sh
    $PY scripts/probe/run_closedloop.py launch --model janus --condition full --gpus 0
    $PY scripts/probe/run_closedloop.py summary --model janus --condition full

Conditions: ``full``, ``drop_video``, ``drop_track``, ``drop_world`` (= Action-only at inference),
``pool_video``, ``swap_same_task``, ``swap_other_task`` (donor K/V from a fixed dataset window of the same /
another task, computed once per task), ``swap_static`` (donor = current observation with the future latents
pinned to the current frame).  Donor swaps replace the future-Video keys only (Track keys, if any, untouched).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_OUT = Path(os.environ.get("PROBE_OUT", "/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes"))
JANUS_P = Path(os.environ.get("JANUS_P", "/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824"))
SIMEVAL = Path("/ytech_milm_intern/danglingwei/outputs/JanusAct4D_imperfect/rt2_v1/simeval_19045")
DEFAULT_TASKS = ["adjust_bottle", "place_empty_cup", "stack_blocks_two", "stack_bowls_two", "handover_block",
                 "click_bell", "open_laptop", "place_a2b_left", "blocks_ranking_rgb", "hanging_mug"]
DENSE = {"stack_bowls_three", "open_microwave", "hanging_mug", "put_bottles_dustbin"}
CONDITIONS = ["full", "drop_video", "drop_track", "drop_world", "pool_video", "swap_same_task", "swap_other_task", "swap_static"]


# ----------------------------------------------------------------------------- policy


def build_policy(model: str, condition: str, windows_path: Path):
    """Constructed inside the GPU worker (imports torch there)."""
    import random
    import torch
    from harness import Intervention, JanusProbe, load_windows
    from janusact4d_rt2imperfect_v1.contracts import scatter_eef80
    from janusact4d_rt2imperfect_v1.data import CAMERAS
    from janusact4d_rt2imperfect_v1.evaluation.policy import eef20_from_observation, eef20_to_sim, images_from_observation
    from openwam.dataloader.robotwin import assemble_multiview_layout
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    probe = JanusProbe() if model == "janus" else __import__("harness_alpha").AlphaProbe()
    windows = [w for w in load_windows(windows_path) if w["kind"] == "event"]
    by_task: dict[str, list] = {}
    for w in windows:
        by_task.setdefault(w["task"], []).append(w)
    tasks_sorted = sorted(by_task)
    donor_cache: dict[str, dict] = {}

    def donor_kv(task: str) -> dict:
        if task not in donor_cache:
            w = by_task[task][0]
            random.seed(w["row"] * 1_000_003 + w["start"] * 7919)
            sample = probe.dataset.read_window(w["row"], w["start"])
            res = probe.predict_window(sample, 4242, capture_kv=True)
            donor_cache[task] = res["kv"]
            print(f"donor K/V cached for {task}: {w['key']} f{w['start']}", flush=True)
        return donor_cache[task]

    class ProbePolicy:
        steps = probe.steps

        def predict(self, request):
            obs, mask, instruction, seed = (request[k] for k in ("observation", "mask", "instruction", "seed"))
            task = request["episode_key"].split("/")[1]
            views = images_from_observation(obs)
            layout = assemble_multiview_layout(views, list(CAMERAS), 384, 320)
            proprio, active = scatter_eef80(probe.normalizer.normalize(eef20_from_observation(obs)))
            sample = dict(
                prompt=format_prompt_for_inference(instruction), first_frame_image=[layout],
                head_rgb=torch.from_numpy(np.asarray(views["head_camera"]).copy()),
                head_depth=torch.from_numpy(np.asarray(obs["observation"]["head_camera"]["depth"], dtype=np.float32)),
                head_mask=torch.from_numpy(np.asarray(mask)),
                proprio=torch.as_tensor(proprio)[None].float() if torch.as_tensor(proprio).ndim == 1 else torch.as_tensor(proprio).float(),
                proprio_mask=torch.as_tensor(active)[None] if torch.as_tensor(active).ndim == 1 else torch.as_tensor(active),
            )
            kw = {}
            if condition == "full":
                pass
            elif condition == "drop_video":
                kw["intervention"] = Intervention("drop", ("video_future",))
            elif condition == "drop_track":
                kw["intervention"] = Intervention("drop", ("track_future",))
            elif condition == "drop_world":
                kw["intervention"] = Intervention("drop")
            elif condition == "pool_video":
                kw["intervention"] = Intervention("pool", ("video_future",))
            elif condition == "swap_same_task":
                kw["intervention"] = Intervention("replace", ("video_future",), donor=donor_kv(task))
            elif condition == "swap_other_task":
                other = tasks_sorted[(tasks_sorted.index(task) + len(tasks_sorted) // 2) % len(tasks_sorted)] if task in tasks_sorted else tasks_sorted[0]
                kw["intervention"] = Intervention("replace", ("video_future",), donor=donor_kv(other))
            elif condition == "swap_static":
                donor = probe.predict_window(sample, seed, capture_kv=True, static_future=True)["kv"]
                kw["intervention"] = Intervention("replace", ("video_future",), donor=donor)
            else:
                raise ValueError(condition)
            save_video = bool(request.get("save_video", True))
            res = probe.predict_window(sample, seed, decode=save_video, **kw)
            values = res["action_norm"]
            native = probe.normalizer.unnormalize(np.concatenate((values[:, :10], values[:, 34:44]), axis=-1))
            actions = eef20_to_sim(native)
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite predicted EEF action")
            out = dict(actions=actions, pred_rgb=None, pred_track=None)
            if save_video:
                out["pred_rgb"] = res["pred_rgb"]
                pt = res.get("pred_track")
                out["pred_track"] = pt if pt is not None and pt.size else np.zeros_like(res["pred_rgb"])
            return out

    return ProbePolicy()


# ----------------------------------------------------------------------------- worker (one per GPU)


def worker_main(output: Path, jobs_path: Path) -> None:
    from janusact4d_rt2imperfect_v1.evaluation import worker as W
    from janusact4d_rt2imperfect_v1.evaluation.campaign import write_json
    manifest = json.loads((output / "campaign.json").read_text())
    jobs = [j for j in json.loads(jobs_path.read_text()) if not (output / j["key"] / "result.json").exists()]
    if not jobs:
        return
    policy = build_policy(manifest["model"], manifest["condition"], Path(manifest["windows"]))
    context = mp.get_context("spawn")
    requests = context.Queue()
    count = min(manifest["sims_per_gpu"], len(jobs))
    responses = [context.Queue() for _ in range(count)]
    processes = [context.Process(target=W.sim_loop, args=(slot, jobs[slot::count], requests, responses[slot], manifest, str(output)))
                 for slot in range(count)]
    for proc in processes:
        proc.start()
    done: set[int] = set()
    try:
        while len(done) < count:
            try:
                slot, request = requests.get(timeout=10)
            except queue.Empty:
                for index, proc in enumerate(processes):
                    if index not in done and proc.exitcode is not None:
                        raise RuntimeError(f"Simulation process {index} exited: {proc.exitcode}")
                continue
            if request is None:
                done.add(slot)
                continue
            try:
                started = time.monotonic()
                prediction = policy.predict(request)
                prediction["inference_seconds"] = time.monotonic() - started
            except Exception:
                traceback.print_exc()
                write_json(output / "logs" / f"inference_error_gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}.json",
                           dict(episode_key=request["episode_key"], error=traceback.format_exc()))
                prediction = dict(error=traceback.format_exc())
            responses[slot].put(prediction)
    finally:
        for proc in processes:
            if len(done) != count and proc.is_alive():
                proc.terminate()
            proc.join()


# ----------------------------------------------------------------------------- launcher / summary


def campaign_dir(a) -> Path:
    return Path(a.out) / "closed" / a.model / a.condition


def launch(a) -> None:
    out = campaign_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((SIMEVAL / "campaign.json").read_text())
    tasks = a.tasks.split(",") if a.tasks else DEFAULT_TASKS
    jobs = [j for j in base["jobs"] if j["task"] in tasks and int(j["episode"]) < a.episodes]
    for j in jobs:
        if j["task"] in DENSE:
            j["execute_steps"] = 8
    manifest = {k: v for k, v in base.items() if k != "jobs"}
    manifest.update(model=a.model, condition=a.condition, windows=str(PROBE_OUT / "windows.jsonl"),
                    sims_per_gpu=a.sims_per_gpu, episode_count=len(jobs), tasks=tasks,
                    robotwin_root=str(JANUS_P / "third_party" / "RoboTwin"))
    (out / "campaign.json").write_text(json.dumps(dict(manifest, jobs=jobs), indent=1))
    pending = [j for j in jobs if not (out / j["key"] / "result.json").exists()]
    if a.limit:
        pending = pending[: a.limit]
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    gpus = a.gpus.split(",")
    procs = []
    for rank, gpu in enumerate(gpus):
        subset = pending[rank::len(gpus)]
        if not subset:
            continue
        job_path = logs / f"jobs_gpu{gpu}.json"
        job_path.write_text(json.dumps(subset))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   VK_ICD_FILENAMES="/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
        log = (logs / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen([sys.executable, "-u", __file__, "worker", "--output", str(out), "--jobs", str(job_path)],
                                cwd=str(JANUS_P), env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append(proc)
    print(f"{len(pending)} episodes pending, {len(procs)} workers -> {out}", flush=True)
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        s = summarize(out)
        print(f"  done {s['complete']}/{s['total']} success {s['success']} ({s['sr']:.1%}) errors {s['error']}", flush=True)
    print(json.dumps(summarize(out), ensure_ascii=False))


def summarize(out: Path) -> dict:
    manifest = json.loads((out / "campaign.json").read_text())
    per_task: dict[str, dict] = {}
    complete = success = error = 0
    for j in manifest["jobs"]:
        p = out / j["key"] / "result.json"
        d = per_task.setdefault(f"{j['phase']}/{j['task']}", dict(n=0, done=0, success=0))
        d["n"] += 1
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r.get("status") == "error":
            error += 1
            continue
        complete += 1
        d["done"] += 1
        success += int(bool(r.get("success")))
        d["success"] += int(bool(r.get("success")))
    s = dict(total=len(manifest["jobs"]), complete=complete, success=success, error=error,
             sr=success / max(complete, 1), per_task=per_task)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="janus", choices=["janus", "alpha"])
        p.add_argument("--condition", default="full", choices=CONDITIONS)
        p.add_argument("--out", default=str(PROBE_OUT))
        p.add_argument("--gpus", default="0")
        p.add_argument("--tasks", default="")
        p.add_argument("--episodes", type=int, default=10, help="episodes per task/phase (max 10 from the frozen protocol)")
        p.add_argument("--sims-per-gpu", type=int, default=6)
        p.add_argument("--limit", type=int, default=0)
    w = sub.add_parser("worker")
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--jobs", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "worker":
        worker_main(a.output, a.jobs)
    elif a.cmd == "launch":
        launch(a)
    else:
        print(json.dumps(summarize(campaign_dir(a)), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
