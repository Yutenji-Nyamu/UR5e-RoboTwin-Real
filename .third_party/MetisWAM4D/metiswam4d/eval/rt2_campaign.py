"""RoboTwin 2.0 closed-loop campaign: 50 tasks x {clean, randomized}, evaluated in rounds of 2 episodes per
task-config (200 episodes per round), so the running numbers after every completed round are balanced.

Seeds: episodes 0-9 of every task-config are the 1000 expert-admitted seeds of the SR93 archive (the cohort of the
JanusAct4D-RT2 evaluation, paired by episode); later episodes scan a disjoint seed block each, after the archive's
``next_seed``, with the official expert admission.  Instructions: seen-language pool of the scene.

Sub-commands (run from the project root with ``/usr/bin/python3.10 -m metiswam4d.eval.rt2_campaign``):
    prepare   freeze ``campaign.json`` (jobs, protocol, model file)
    run       one manager per host: one model worker per GPU, each serving ``--sims`` simulation processes that
              claim the next unfinished episode from the shared job list (claims are exclusive files on the
              shared disk, so several hosts drain the same campaign)
    summary   ``summary.json`` / ``summary.md`` from the episode results

Layout: ``<out>/episodes/<phase>/<task>/episode_NNN/{result.json, scene.json, chunks.jsonl, simulation.log,
closed_loop.mp4}``; ``<out>/claims/``; ``<out>/logs/``.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

ARCHIVE = Path("/ytech_milm_intern/danglingwei/outputs/JanusTrack4d_260824/V3_step30k_SR93/evaluation_evidence")
PAIRED = Path("/ytech_milm_intern/danglingwei/outputs/JanusAct4D_imperfect/rt2_v1/simeval_19045")
PHASES = {"clean": "demo_clean", "random": "demo_randomized"}
DENSE_TASKS = ("stack_bowls_three", "open_microwave", "hanging_mug", "put_bottles_dustbin")
EPISODES_PER_ROUND = 2
SEED_BLOCK = 100
VIDEO_EPISODES = (0,)
BASELINES = {  # micro-SR % (clean, random, average), from the evaluation records
    "JanusAct4D-RT2-ipft": (96.52, 90.38, 93.46),
    "JanusTrack4D-v3": (93.20, 93.00, 93.10),
    "OpenWAM-alpha (paper)": (92.84, 92.36, 92.60),
    "FastWAM": (86.11, 85.56, 85.83),
    "Efficient-WAM": (80.44, 80.56, 80.50),
}


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def episode_dir(out: Path, job: dict) -> Path:
    return out / "episodes" / job["key"]


# ----------------------------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------------------------

def prepare(out: Path, config: str, model_file: str, source_checkpoint: str, rounds: int, execute_steps: int,
            dense_steps: int, sims_per_gpu: int) -> dict:
    paired = json.loads((PAIRED / "campaign.json").read_text())["jobs"]
    next_seed: dict[tuple[str, str], int] = {}
    for path in glob.glob(str(ARCHIVE / "*" / "*" / "*" / "progress.json")):
        data = json.loads(Path(path).read_text())
        group = (data["task"], data["phase"])
        next_seed[group] = max(next_seed.get(group, 0), int(data["next_seed"]))
    archived = {(j["task"], j["phase"], int(j["episode"])): j for j in paired}
    tasks = sorted({j["task"] for j in paired})
    jobs = []
    for episode in range(rounds * EPISODES_PER_ROUND):
        for task in tasks:
            for phase in PHASES:
                job = {"task": task, "phase": phase, "task_config": PHASES[phase], "episode": episode,
                       "round": episode // EPISODES_PER_ROUND, "key": f"{phase}/{task}/episode_{episode:03d}",
                       "execute_steps": dense_steps if task in DENSE_TASKS else execute_steps}
                src = archived.get((task, phase, episode))
                if src is not None:
                    job.update(environment_seed=int(src["environment_seed"]),
                               protocol_episode_index=int(src["protocol_episode_index"]),
                               fallback_instruction=src["instruction"])
                else:
                    job.update(environment_seed=None, protocol_episode_index=episode,
                               seed_block=next_seed[(task, phase)] + (episode - 10) * SEED_BLOCK,
                               block_size=SEED_BLOCK)
                jobs.append(job)
    # Seen instructions already resolved for the same seed by the paired evaluation (same generator, same seed).
    reused = 0
    for job in jobs:
        if job["episode"] >= 10:
            continue
        old = PAIRED / job["key"].replace(f"episode_{job['episode']:03d}", f"episode_{job['episode']:02d}")
        inst, res = old / "instruction.json", old / "result.json"
        if inst.exists() and res.exists():
            info, result = json.loads(inst.read_text()), json.loads(res.read_text())
            if info.get("instruction_type") == "seen" and result.get("environment_seed") is not None:
                write_json(episode_dir(out, job) / "scene.json", {
                    "environment_seed": int(result["environment_seed"]), "info": info.get("scene_info"),
                    "expert_ok": True, "instruction": info["instruction"], "instruction_type": "seen",
                    "source": str(inst)})
                reused += 1
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": str(Path(config).resolve()),
        "model_file": model_file, "source_checkpoint": source_checkpoint,
        "protocol": {"instruction_type": "seen", "rounds": 20, "stop_after": "action (round 10)",
                     "execute_steps": execute_steps, "dense_tasks": list(DENSE_TASKS), "dense_execute_steps": dense_steps,
                     "episodes_per_round": EPISODES_PER_ROUND, "seed_source": "SR93 archive episodes 0-9, "
                     f"then expert-admitted seeds in disjoint blocks of {SEED_BLOCK} after the archive next_seed",
                     "policy_seed": "sha256(metiswam4d-rt2:environment_seed:episode:replan)",
                     "video_episodes": list(VIDEO_EPISODES)},
        "sims_per_gpu": sims_per_gpu, "seen_instructions_reused": reused, "jobs": jobs,
    }
    write_json(out / "campaign.json", manifest)
    return manifest


STAGE_C = Path("/ytech_milm_intern/danglingwei/outputs/JanusTrack4d_260824/evaluation/fusion_v3_step30000_stage_c_frozen300_k3_v1")
STAGE_C_SCENES = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/rt2_direct_v3_coupled/simeval_33760/"
                      "ablation_dense_exec/episodes/e24")
DEV_DENSE_TASKS = ("stack_bowls_three", "hanging_mug", "put_bottles_dustbin")  # the campaign routing (execute 8)


def prepare_dev(out: Path, config: str, model_file: str, source_checkpoint: str, tasks: list[str],
                variants: dict[str, dict], execute_steps: int, dense_steps: int, sims_per_gpu: int) -> dict:
    """Inference-variant comparison on the Stage C seeds (the old project's independent confirmation set, disjoint
    from the campaign seeds).  Every variant runs every seed (paired); round = seed index, so a complete round is
    one seed of every task-config under every variant."""
    jobs = []
    for task in tasks:
        for phase in PHASES:
            rows = [json.loads(l) for l in (STAGE_C / phase / task / "episodes.jsonl").read_text().splitlines() if l.strip()]
            for i, row in enumerate(rows):
                episode = 100 + i
                for name, policy in variants.items():
                    job = {"task": task, "phase": phase, "task_config": PHASES[phase], "episode": episode, "round": i,
                           "key": f"{name}/{phase}/{task}/episode_{episode:03d}", "variant": name, "policy": policy,
                           "execute_steps": dense_steps if task in DEV_DENSE_TASKS else execute_steps,
                           "environment_seed": int(row["environment_seed"]),
                           "protocol_episode_index": int(row["protocol_episode_index"]),
                           "fallback_instruction": row["instruction"]}
                    jobs.append(job)
                    scene = STAGE_C_SCENES / phase / task / f"episode_{episode:03d}" / "scene.json"
                    target = episode_dir(out, job) / "scene.json"
                    if scene.exists() and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(scene.read_text())
    jobs.sort(key=lambda j: (j["round"], j["task"], j["phase"], j["variant"]))
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": str(Path(config).resolve()),
        "model_file": model_file, "source_checkpoint": source_checkpoint,
        "protocol": {"purpose": "inference variants on the weak tasks, Stage C seeds (disjoint from the campaign)",
                     "instruction_type": "seen", "execute_steps": execute_steps, "dense_tasks": list(DEV_DENSE_TASKS),
                     "dense_execute_steps": dense_steps, "variants": variants,
                     "policy_seed": "sha256(metiswam4d-rt2:environment_seed:episode:replan); sample k>0: sha256(seed:k)"},
        "sims_per_gpu": sims_per_gpu, "jobs": jobs,
    }
    write_json(out / "campaign.json", manifest)
    return manifest


OFFICIAL_SEED_START = 100000   # eval_policy.py: st_seed = 100000 * (1 + seed), seed 0
OFFICIAL_TEST_NUM = 100        # admitted seeds evaluated per task-config
OFFICIAL_REFERENCE = {  # micro-SR % (clean, random, average)
    "OpenWAM-alpha (official README)": (93.74, 93.46, 93.60),
    "OpenWAM-alpha (cited before)": (92.84, 92.36, 92.60),
    "MetisWAM4D avg5 (our seen protocol)": (93.50, 92.75, 93.13),
}


def prepare_official(out: Path, model_kind: str, model_file: str, source_checkpoint: str, label: str,
                     tasks: list[str] | None, offsets: int, instruction_type: str, execute_steps: int,
                     sims_per_gpu: int) -> dict:
    """The official RoboTwin protocol as a campaign: one job per (task-config, seed) for the seeds
    ``100000 .. 100000 + offsets - 1``, round = seed offset.  A job runs the official expert admission and, if the seed
    is admitted, the policy episode on it; a task-config is scored on its first ``OFFICIAL_TEST_NUM`` admitted seeds
    and stops handing out jobs beyond them (``closed.json``)."""
    if not tasks:
        tasks = sorted({j["task"] for j in json.loads((PAIRED / "campaign.json").read_text())["jobs"]})
    jobs = []
    for offset in range(offsets):
        seed = OFFICIAL_SEED_START + offset
        for task in tasks:
            for phase in PHASES:
                jobs.append({"task": task, "phase": phase, "task_config": PHASES[phase], "episode": offset,
                             "round": offset, "key": f"{phase}/{task}/seed_{seed}", "environment_seed": seed,
                             "protocol_episode_index": offset, "admission": "official",
                             "instruction_type": instruction_type, "execute_steps": execute_steps})
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "model_kind": model_kind, "label": label, "config": "",
        "model_file": model_file, "source_checkpoint": source_checkpoint,
        "protocol": {"admission": "official", "seed_start": OFFICIAL_SEED_START, "test_num": OFFICIAL_TEST_NUM,
                     "instruction_type": instruction_type, "instruction_pool": "generate_episode_descriptions(task, "
                     "[info], 100)[type], pick = sha256(seed)", "execute_steps": execute_steps,
                     "step_limits": "task_config/_eval_step_limit.yml", "video_episodes": list(VIDEO_EPISODES)},
        "sims_per_gpu": sims_per_gpu, "jobs": jobs,
    }
    write_json(out / "campaign.json", manifest)
    return manifest


# ----------------------------------------------------------------------------------------------------------------
# claims
# ----------------------------------------------------------------------------------------------------------------

def claim_path(out: Path, job: dict) -> Path:
    return out / "claims" / (job["key"].replace("/", "__") + ".claim")


def try_claim(out: Path, job: dict, tag: str) -> bool:
    path = claim_path(out, job)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"tag": tag, "pid": os.getpid(), "time": time.time()}))
    return True


def release_stale_claims(out: Path, jobs: list[dict], prefix: str) -> int:
    """Drop claims held by (dead) simulation processes whose tag starts with ``prefix`` and have no result."""
    released = 0
    for job in jobs:
        path = claim_path(out, job)
        if not path.exists() or (episode_dir(out, job) / "result.json").exists():
            continue
        try:
            tag = json.loads(path.read_text()).get("tag", "")
        except (OSError, ValueError):
            tag = ""
        if tag.startswith(prefix):
            path.unlink(missing_ok=True)
            released += 1
    return released


# ----------------------------------------------------------------------------------------------------------------
# simulation process
# ----------------------------------------------------------------------------------------------------------------

def _labeled(frame: np.ndarray, label: str, size=(320, 384)) -> np.ndarray:
    import cv2
    scale = min(size[0] / frame.shape[1], size[1] / frame.shape[0])
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (size[0], 22), (0, 0, 0), -1)
    cv2.putText(canvas, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return canvas


def run_episode(job: dict, folder: Path, infer) -> dict:
    from metiswam4d.data.rt2.text_cache import format_prompt
    from metiswam4d.eval.rt2_sim import SeedRejected, close_env, create_env, observe, policy_seed, resolve_scene
    alpha = job.get("model_kind") == "openwam_alpha"
    if alpha:
        from metiswam4d.eval.alpha_policy import alpha_observe
    result = dict(job, success=False, status="running", replans=0, final_step=0, model_started=False)
    started = time.monotonic()
    timing = {"scene": 0.0, "setup": 0.0, "wait": 0.0, "act": 0.0}
    env = writer = chunks = None
    try:
        tick = time.monotonic()
        scene = resolve_scene(job, folder)
        seed = int(scene["environment_seed"])
        result.update(environment_seed_used=seed, instruction=scene["instruction"],
                      instruction_type=scene["instruction_type"])
        timing["scene"] = time.monotonic() - tick
        tick = time.monotonic()
        env, fg_ids = create_env(job["task"], job["task_config"], seed, int(job["protocol_episode_index"]))
        timing["setup"] = time.monotonic() - tick
        plan = {"calls": 0, "fail": 0}
        for side in ("left", "right"):  # cuRobo outcome of every executed action (a failed plan leaves the arm still)
            def counted(*a, _orig=getattr(env.robot, f"{side}_plan_path"), **k):
                out = _orig(*a, **k)
                plan["calls"] += 1
                plan["fail"] += out.get("status") != "Success"
                return out
            setattr(env.robot, f"{side}_plan_path", counted)
        result["plan"] = plan
        env.set_instruction(instruction=scene["instruction"])
        result["step_limit"] = int(env.step_lim)
        save_video = int(job["episode"]) in VIDEO_EPISODES
        if save_video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(folder / "closed_loop.mp4"), fps=8, codec="libx264", macro_block_size=1,
                                        ffmpeg_params=["-crf", "22", "-threads", "2"])
        prompt = format_prompt(scene["instruction"])
        obs = env.get_obs()
        chunks = open(folder / "chunks.jsonl", "w")
        result["model_started"] = True
        while env.take_action_cnt < env.step_lim and not env.eval_success:
            if alpha:
                request = dict(alpha_observe(obs), instruction=scene["instruction"], full=save_video)
            else:
                request = observe(env, obs, fg_ids)
                request.update(prompt=prompt, seed=policy_seed(seed, int(job["episode"]), result["replans"]),
                               full=save_video, policy=job.get("policy") or {})
            tick = time.monotonic()
            pred = infer(request)
            wait = time.monotonic() - tick
            timing["wait"] += wait
            tick = time.monotonic()
            first_step = int(env.take_action_cnt)
            actual = [obs["observation"]["head_camera"]["rgb"]] if save_video else None
            executed = 0
            for action in pred["actions"][: int(job["execute_steps"])]:
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
                env.take_action(action, action_type="ee")
                executed += 1
                if save_video:
                    actual.append(env.get_obs()["observation"]["head_camera"]["rgb"])
            obs = env.get_obs()
            timing["act"] += time.monotonic() - tick
            if save_video:
                for k in range(9):
                    t = min(4 * k, executed)
                    writer.append_data(np.concatenate([
                        _labeled(actual[t], f"ACTUAL step {first_step + t}" + (" (held)" if 4 * k > executed else "")),
                        _labeled(pred["pred_video"][k], f"PRED VIDEO +{4 * k}"),
                        _labeled(pred["pred_track"][k], f"PRED TRACK +{4 * k}")], axis=1))
            chunks.write(json.dumps({"replan": result["replans"], "start_step": first_step, "executed": executed,
                                     "wait_seconds": round(wait, 3), "inference_seconds": round(pred["inference_seconds"], 3),
                                     "batch": pred.get("batch"), "sample_spread": round(pred.get("sample_spread", 0.0), 4),
                                     "eef20": np.round(pred["eef20"], 5).tolist()}) + "\n")
            chunks.flush()
            result["replans"] += 1
            result["final_step"] = int(env.take_action_cnt)
        result.update(success=bool(env.eval_success), status="complete", final_step=int(env.take_action_cnt))
    except SeedRejected as exc:
        result.update(status="rejected", note=str(exc))
    except Exception:
        result.update(status="error", error=traceback.format_exc()[-3000:])
        traceback.print_exc()
    finally:
        for handle in (writer, chunks):
            if handle is not None:
                handle.close()
        if env is not None:
            for side in ("left", "right"):
                env.robot.__dict__.pop(f"{side}_plan_path", None)
            close_env(env)
        result["elapsed_seconds"] = round(time.monotonic() - started, 1)
        result["timing"] = {k: round(v, 1) for k, v in timing.items()}
    return result


def pending_jobs(outs: list[Path]):
    """``(out, job)`` in campaign order that have neither a claim nor a result; ``deferred`` jobs (inference routing
    not decided yet) are not handed out.  The manifests are re-read on every call, so edits to ``campaign.json``
    (un-deferring a task, changing its policy) take effect at the next episode of every simulation process."""
    for out in outs:
        claimed = set(os.listdir(out / "claims")) if (out / "claims").exists() else set()
        manifest = json.loads((out / "campaign.json").read_text())
        kind = manifest.get("model_kind", "metiswam4d")
        closed = json.loads((out / "closed.json").read_text()) if (out / "closed.json").exists() else {}
        for job in manifest["jobs"]:
            if job.get("deferred") or claim_path(out, job).name in claimed:
                continue
            if job["round"] > closed.get(f"{job['phase']}/{job['task']}", float("inf")):
                continue
            if not (episode_dir(out, job) / "result.json").exists():
                yield out, dict(job, model_kind=kind)


def next_job(outs: list[Path], tag: str):
    """Claim the earliest pending job (campaign order = round by round), or ``None`` when everything is taken."""
    for out, job in pending_jobs(outs):
        if try_claim(out, job, tag):
            return out, job
    return None


def sim_loop(slot: int, tag: str, outs: list[str], requests, responses) -> None:
    from metiswam4d.eval.rt2_sim import setup_runtime
    setup_runtime()

    def infer(request: dict) -> dict:
        requests.put((slot, request))
        reply = responses.get()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    while True:
        claimed = next_job([Path(o) for o in outs], tag)
        if claimed is None:
            break
        out, job = claimed
        folder = episode_dir(out, job)
        folder.mkdir(parents=True, exist_ok=True)
        log = open(folder / "simulation.log", "a", buffering=1)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log
        try:
            result = run_episode(job, folder, infer)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            log.close()
        result["worker"] = tag
        if result["status"] == "error" and not result["model_started"]:  # a rejected seed is a final result
            # Failed before the policy ran (typically the renderer of this process: "cannot create buffer"): keep the
            # record, hand the episode back and restart the process; the third such failure is final.
            attempts = len(list(folder.glob("error_*.json")))
            if attempts < 2:
                write_json(folder / f"error_{attempts}.json", result)
                claim_path(out, job).unlink(missing_ok=True)
                print(json.dumps({"key": job["key"], "status": "retry", "error": result["error"].strip().splitlines()[-1]}),
                      flush=True)
                os._exit(3)
        write_json(folder / "result.json", result)
        print(json.dumps({k: result.get(k) for k in ("key", "success", "status", "final_step", "replans",
                                                       "elapsed_seconds")}), flush=True)
    requests.put((slot, None))


# ----------------------------------------------------------------------------------------------------------------
# model worker (one per GPU)
# ----------------------------------------------------------------------------------------------------------------

def worker(outs: list[Path], tag: str, sims: int, max_batch: int) -> None:
    """One model for all campaigns in ``outs`` (drained in order; they must share the model file)."""
    import torch
    from metiswam4d.eval.rt2_policy import RT2Policy
    manifests = [json.loads((out / "campaign.json").read_text()) for out in outs]
    if len({(m["config"], m["model_file"]) for m in manifests}) != 1:
        raise ValueError("campaigns served by one worker must share config and model file")

    def release(prefix: str) -> int:
        return sum(release_stale_claims(out, m["jobs"], prefix) for out, m in zip(outs, manifests))

    print(f"[{tag}] released {release(tag + '-')} stale claims", flush=True)
    torch.set_num_threads(4)
    log = lambda m: print(f"[{tag}] {m}", flush=True)
    if manifests[0].get("model_kind") == "openwam_alpha":
        from metiswam4d.eval.alpha_policy import AlphaRT2Policy
        policy = AlphaRT2Policy(manifests[0]["model_file"], log=log)
    else:
        policy = RT2Policy(manifests[0]["config"], manifests[0]["model_file"], log=log)
    ctx = mp.get_context("spawn")
    requests = ctx.Queue()
    responses = [ctx.Queue() for _ in range(sims)]

    def start(slot: int):
        proc = ctx.Process(target=sim_loop, args=(slot, f"{tag}-s{slot}", [str(o) for o in outs], requests,
                                                  responses[slot]))
        proc.start()
        return proc

    procs = [start(s) for s in range(sims)]
    done: set[int] = set()
    while len(done) < sims:
        for s, proc in enumerate(procs):
            if s not in done and proc.exitcode is not None:
                print(f"[{tag}] sim {s} exited ({proc.exitcode}); restarting", flush=True)
                release(f"{tag}-s{s}")
                procs[s] = start(s)
        try:
            batch = [requests.get(timeout=10)]
        except queue.Empty:
            continue
        while len(batch) < max_batch:
            try:
                batch.append(requests.get_nowait())
            except queue.Empty:
                break
        live = []
        for slot, request in batch:
            if request is None:
                done.add(slot)
            else:
                live.append((slot, request))
        groups = collections.defaultdict(list)
        for slot, request in live:
            groups[(bool(request.get("full")), json.dumps(request.get("policy") or {}, sort_keys=True))].append(
                (slot, request))
        for group in groups.values():
            try:
                preds = policy.predict_batch([r for _, r in group])
                for (slot, _), pred in zip(group, preds):
                    responses[slot].put(pred)
            except Exception:
                err = traceback.format_exc()
                print(f"[{tag}] inference error\n{err}", flush=True)
                for slot, _ in group:
                    responses[slot].put({"error": err[-2000:]})
    for proc in procs:
        proc.join()


# ----------------------------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------------------------

def _rates(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["success"]]
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["success"])
    return {"episodes": len(rows), "successes": len(ok),
            "micro_sr": 100.0 * len(ok) / len(rows) if rows else None,
            "macro_sr": 100.0 * float(np.mean([np.mean(v) for v in by_task.values()])) if rows else None,
            "success_steps": float(np.mean([r["final_step"] for r in ok])) if ok else None}


def summarize(out: Path) -> dict:
    """An episode that fails after the policy started (simulator / planner exception) counts as a failure; one that
    never reached the policy (no admissible scene) is listed under ``errors`` and left out of the denominator."""
    manifest = json.loads((out / "campaign.json").read_text())
    if manifest.get("protocol", {}).get("admission") == "official":
        return summarize_official(out, manifest)
    results, errors, settled = {}, [], set()
    for job in manifest["jobs"]:
        path = episode_dir(out, job) / "result.json"
        if path.exists():
            r = json.loads(path.read_text())
            settled.add(job["key"])
            if r.get("status") == "complete" or r.get("model_started"):
                results[job["key"]] = r
            if r.get("status") != "complete":
                errors.append({"key": job["key"], "model_started": r.get("model_started"),
                               "error": (r.get("error") or "")[-300:]})
    by_round = collections.defaultdict(list)
    for job in manifest["jobs"]:
        by_round[job["round"]].append(job["key"])
    complete_rounds = 0
    for rnd in sorted(by_round):
        if all(k in settled for k in by_round[rnd]):
            complete_rounds = rnd + 1
        else:
            break
    def block(rows):
        return {"clean": _rates([r for r in rows if r["phase"] == "clean"]),
                "random": _rates([r for r in rows if r["phase"] == "random"]), "average": _rates(rows)}
    balanced = [r for r in results.values() if r["round"] < complete_rounds]
    per_task = collections.defaultdict(lambda: {"clean": [0, 0], "random": [0, 0]})
    for r in results.values():
        cell = per_task[r["task"]][r["phase"]]
        cell[0] += int(r["success"]); cell[1] += 1
    summary = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "source_checkpoint": manifest["source_checkpoint"],
               "complete_rounds": complete_rounds, "episodes_per_task_config": complete_rounds * EPISODES_PER_ROUND,
               "balanced": block(balanced), "all_finished": block(list(results.values())),
               "finished": len(results), "errors": errors, "per_task": dict(sorted(per_task.items()))}
    if any("variant" in job for job in manifest["jobs"]):
        by_variant = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for r in results.values():
            for cell in (by_variant[r["variant"]][r["task"]], by_variant[r["variant"]]["all"]):
                cell[0] += int(r["success"]); cell[1] += 1
        summary["by_variant"] = {v: dict(sorted(t.items())) for v, t in by_variant.items()}
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(summary))
    return summary


def summarize_official(out: Path, manifest: dict) -> dict:
    """Official-protocol scores.  Per task-config the seeds are walked in order; a seed counts once every earlier
    seed of that config is settled, and the config is closed at its ``test_num``-th admitted seed (later seeds are
    surplus).  A seed whose job ended with a final error before the policy started is skipped like a rejection and
    listed under ``errors``.  ``complete_rounds`` R: every config has settled all its seeds below R (or is closed);
    the balanced numbers use the admitted seeds below R."""
    test_num = int(manifest["protocol"]["test_num"])
    by_config = collections.defaultdict(list)
    for job in manifest["jobs"]:
        by_config[(job["phase"], job["task"])].append(job)
    configs = {}
    errors = []
    for (phase, task), jobs in sorted(by_config.items()):
        jobs.sort(key=lambda j: j["round"])
        rows, rejected, frontier, cutoff = [], 0, len(jobs), None
        for i, job in enumerate(jobs):
            path = episode_dir(out, job) / "result.json"
            if not path.exists():
                frontier = job["round"]
                break
            r = json.loads(path.read_text())
            if r["status"] == "rejected":
                rejected += 1
                continue
            if r["status"] != "complete" and not r.get("model_started"):
                errors.append({"key": job["key"], "error": (r.get("error") or "")[-300:]})
                continue
            rows.append(r)
            if len(rows) == test_num:
                cutoff = job["round"]
                frontier = cutoff + 1
                break
        else:
            frontier = jobs[-1]["round"] + 1
        configs[(phase, task)] = {"rows": rows, "rejected": rejected, "frontier": frontier, "cutoff": cutoff,
                                  "exhausted": cutoff is None and frontier > jobs[-1]["round"]}
    closed = {f"{p}/{t}": c["cutoff"] for (p, t), c in configs.items() if c["cutoff"] is not None}
    write_json(out / "closed.json", closed)
    complete_rounds = min(c["frontier"] for c in configs.values())
    balanced = [r for c in configs.values() for r in c["rows"] if r["round"] < complete_rounds]
    every = [r for c in configs.values() for r in c["rows"]]

    def block(rows):
        return {"clean": _rates([r for r in rows if r["phase"] == "clean"]),
                "random": _rates([r for r in rows if r["phase"] == "random"]), "average": _rates(rows)}
    per_task = collections.defaultdict(dict)
    for (phase, task), c in configs.items():
        per_task[task][phase] = {"success": sum(int(r["success"]) for r in c["rows"]), "evaluated": len(c["rows"]),
                                 "rejected": c["rejected"], "next_seed": OFFICIAL_SEED_START + c["frontier"],
                                 "closed": c["cutoff"] is not None, "exhausted": c["exhausted"]}
    summary = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "label": manifest.get("label"),
               "source_checkpoint": manifest["source_checkpoint"], "protocol": manifest["protocol"],
               "complete_rounds": complete_rounds, "closed_configs": len(closed), "configs": len(configs),
               "balanced": block(balanced), "evaluated": block(every), "errors": errors,
               "exhausted": sorted(f"{p}/{t}" for (p, t), c in configs.items() if c["exhausted"]),
               "per_task": dict(sorted(per_task.items()))}
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(official_markdown(summary))
    return summary


def official_markdown(s: dict) -> str:
    fmt = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    p = s["protocol"]
    lines = [f"# RoboTwin 2.0 official protocol — {s['label']}", "",
             f"Updated {s['updated']}.  Seeds from {p['seed_start']} with the expert admission, {p['test_num']} "
             f"admitted seeds per task-config, {p['instruction_type']} instructions, execute {p['execute_steps']} "
             f"steps per chunk.  Closed task-configs: **{s['closed_configs']}/{s['configs']}**; complete seed rounds: "
             f"**{s['complete_rounds']}**; errors {len(s['errors'])}.", "",
             "| Config | Balanced (seeds < complete rounds) | macro | success-steps | All evaluated | "
             + " | ".join(OFFICIAL_REFERENCE) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(OFFICIAL_REFERENCE)]
    for i, phase in enumerate(("clean", "random", "average")):
        b, e = s["balanced"][phase], s["evaluated"][phase]
        lines.append(f"| {phase} | {fmt(b['micro_sr'])}% ({b['successes']}/{b['episodes']}) | {fmt(b['macro_sr'])}% | "
                     f"{fmt(b['success_steps'], 1)} | {fmt(e['micro_sr'])}% ({e['successes']}/{e['episodes']}) | "
                     + " | ".join(f"{v[i]:.2f}%" for v in OFFICIAL_REFERENCE.values()) + " |")
    if s["exhausted"]:
        lines += ["", "Seed list exhausted before 100 admitted seeds: " + ", ".join(s["exhausted"])]
    lines += ["", "| Task | clean | random | rejected seeds (clean / random) |", "|---|---:|---:|---:|"]
    for task, cells in s["per_task"].items():
        c, r = cells.get("clean", {}), cells.get("random", {})
        lines.append(f"| {task} | {c.get('success', 0)}/{c.get('evaluated', 0)} | {r.get('success', 0)}/"
                     f"{r.get('evaluated', 0)} | {c.get('rejected', 0)} / {r.get('rejected', 0)} |")
    return "\n".join(lines) + "\n"


def summary_markdown(s: dict) -> str:
    fmt = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    b = s["balanced"]
    lines = [f"# RT2 closed loop — {Path(s['source_checkpoint']).name}", "",
             f"Updated {s['updated']}.  Complete rounds: **{s['complete_rounds']}** "
             f"({s['episodes_per_task_config']} episodes per task-config); finished episodes {s['finished']}, "
             f"errors {len(s['errors'])}.", "",
             "| Config | Metric | MetisWAM4D | " + " | ".join(BASELINES) + " |",
             "|---|---|---:|" + "---:|" * len(BASELINES)]
    for i, phase in enumerate(("clean", "random", "average")):
        r = b[phase]
        count = f" ({r['successes']}/{r['episodes']})" if r["episodes"] else ""
        lines.append(f"| {phase.capitalize()} | micro-sr | {fmt(r['micro_sr'])}%{count} | "
                     + " | ".join(f"{v[i]:.2f}%" for v in BASELINES.values()) + " |")
        lines.append(f"|  | macro-sr | {fmt(r['macro_sr'])}% |" + " |" * len(BASELINES))
        lines.append(f"|  | success-steps | {fmt(r['success_steps'])} |" + " |" * len(BASELINES))
    a = s["all_finished"]["average"]
    lines += ["", f"All finished episodes (incl. the running round): {a['successes']}/{a['episodes']} = "
              f"{fmt(a['micro_sr'])}%.", "", "| Task | clean | random |", "|---|---:|---:|"]
    for task, cell in s["per_task"].items():
        lines.append(f"| {task} | {cell['clean'][0]}/{cell['clean'][1]} | {cell['random'][0]}/{cell['random'][1]} |")
    if s.get("by_variant"):
        variants = sorted(s["by_variant"])
        tasks = sorted({t for v in variants for t in s["by_variant"][v]} - {"all"}) + ["all"]
        lines += ["", "## By variant (clean + random)", "", "| Task | " + " | ".join(variants) + " |",
                  "|---|" + "---:|" * len(variants)]
        for task in tasks:
            cells = [s["by_variant"][v].get(task, [0, 0]) for v in variants]
            lines.append(f"| {task} | " + " | ".join(f"{a}/{n}" for a, n in cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# manager
# ----------------------------------------------------------------------------------------------------------------

def run(outs: list[Path], gpus: list[str], sims: int, max_batch: int) -> None:
    host = socket.gethostname().split(".")[0]
    for out in outs:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "claims").mkdir(exist_ok=True)
    logs = outs[0] / "logs"
    project = Path(__file__).resolve().parents[2]
    procs = []
    for gpu in gpus:
        tag = f"{host}-gpu{gpu}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(project), OMP_NUM_THREADS="4",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   VK_ICD_FILENAMES="/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json")
        log = open(logs / f"{tag}.log", "a")
        procs.append((subprocess.Popen([sys.executable, "-u", "-m", "metiswam4d.eval.rt2_campaign", "worker",
                                        "--output", ",".join(map(str, outs)), "--tag", tag, "--sims", str(sims),
                                        "--max-batch", str(max_batch)],
                                       cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    write_json(logs / f"manager_{host}.json", {"pid": os.getpid(), "workers": [p.pid for p, _ in procs],
                                                "gpus": gpus, "sims": sims, "started": time.strftime("%F %T")})
    while any(p.poll() is None for p, _ in procs):
        for out in outs:
            try:
                summarize(out)
            except Exception:
                traceback.print_exc()
        time.sleep(300)
    for out in outs:
        summarize(out)
    for _, log in procs:
        log.close()
    if any(p.returncode for p, _ in procs):
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "prepare-dev", "prepare-official", "run", "worker", "summary"])
    ap.add_argument("--output", required=True, help="campaign directory (run / worker: comma-separated, drained in order)")
    ap.add_argument("--tasks", help="prepare-dev / prepare-official: comma-separated tasks")
    ap.add_argument("--model-kind", default="openwam_alpha", help="prepare-official: openwam_alpha | metiswam4d")
    ap.add_argument("--label", default="OpenWAM-Alpha-Sim-RoboTwin-Full", help="prepare-official: model name")
    ap.add_argument("--offsets", type=int, default=200, help="prepare-official: seeds 100000 .. 100000+offsets-1")
    ap.add_argument("--instruction-type", default="unseen", help="prepare-official: seen | unseen")
    ap.add_argument("--variants", help="prepare-dev: JSON {name: policy options}")
    ap.add_argument("--config", default="configs/rt2_direct_v3_coupled.yaml")
    ap.add_argument("--model-file")
    ap.add_argument("--source-checkpoint")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--execute-steps", type=int, default=24)
    ap.add_argument("--dense-steps", type=int, default=8)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--sims", type=int, default=6)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--tag")
    args = ap.parse_args()
    outs = [Path(p) for p in args.output.split(",")]
    if args.mode == "prepare":
        m = prepare(outs[0], args.config, args.model_file, args.source_checkpoint, args.rounds,
                    args.execute_steps, args.dense_steps, args.sims)
        print(json.dumps({"jobs": len(m["jobs"]), "seen_instructions_reused": m["seen_instructions_reused"]}))
    elif args.mode == "prepare-dev":
        m = prepare_dev(outs[0], args.config, args.model_file, args.source_checkpoint, args.tasks.split(","),
                        json.loads(args.variants), args.execute_steps, args.dense_steps, args.sims)
        print(json.dumps({"jobs": len(m["jobs"])}))
    elif args.mode == "prepare-official":
        m = prepare_official(outs[0], args.model_kind, args.model_file, args.source_checkpoint or args.model_file,
                             args.label, args.tasks.split(",") if args.tasks else None, args.offsets,
                             args.instruction_type, args.execute_steps, args.sims)
        print(json.dumps({"jobs": len(m["jobs"])}))
    elif args.mode == "run":
        run(outs, args.gpus.split(","), args.sims, args.max_batch)
    elif args.mode == "worker":
        worker(outs, args.tag, args.sims, args.max_batch)
    else:
        s = summarize(outs[0])
        print(json.dumps({k: s[k] for k in ("complete_rounds", "finished", "balanced")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
