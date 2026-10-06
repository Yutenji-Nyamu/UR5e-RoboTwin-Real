"""XPolicyLab ``deploy`` hook of the scripted expert (inside the official RoboDojo client, Isaac, Python 3.11).

Same cadence and recorder as the InternW0 teacher (``iw0_deploy``): one ``take_action`` per command, ``get_obs`` after
every action, ``iw0_deploy.Recorder4D`` writing ``<METIS_RECORD_DIR>/<variant>/layout_NNN.hdf5`` (plus the head-camera
ground truth with ``METIS_RECORD_4D=1``).  The commands come from ``expert.<task>.episode(scene)``; when the expert
gives up before the step limit (or raises) the episode is closed as a failure.  One JSON line per episode goes to
``METIS_EXPERT_LOG``.  ``METIS_EXPERT_RELOAD=1`` re-imports the expert modules every episode (development);
``METIS_EXPERT_PROBE=1`` runs the task module's ``probe`` before the first episode's commands.
"""
import importlib
import json
import os
import time
import traceback

import numpy as np

from metiswam4d_inspired_by_internw0.eval import iw0_deploy
from metiswam4d_inspired_by_internw0.expert import skills
from metiswam4d_inspired_by_internw0.joint_action import joints_to_action_dicts

PACKAGE = "metiswam4d_inspired_by_internw0.expert"
RELOAD = os.environ.get("METIS_EXPERT_RELOAD") == "1"
PROBE = os.environ.get("METIS_EXPERT_PROBE") == "1"
LOG = os.environ.get("METIS_EXPERT_LOG", "")
STATE = {"episodes": 0}


def expert_module(task: str):
    if RELOAD:
        importlib.invalidate_caches()
    module = importlib.import_module(f"{PACKAGE}.{task}")
    if RELOAD:
        importlib.reload(skills)
        importlib.reload(importlib.import_module(f"{PACKAGE}.grasping"))
        module = importlib.reload(module)
    return module


def eval_one_episode(TASK_ENV, model_client):
    variant = os.environ.get("METIS_VARIANT", TASK_ENV.task_name)
    layout = int(TASK_ENV.env_seeds[0])
    started = time.time()
    recorder = iw0_deploy.Recorder4D(TASK_ENV, variant, layout) if iw0_deploy.RECORD_DIR else None
    if recorder is not None:
        recorder.add(recorder.head)
    else:
        TASK_ENV.get_obs()
    notes: list[str] = []
    error = None
    steps = 0

    def commands():
        module = expert_module(TASK_ENV.task_name)
        scene = skills.Scene(TASK_ENV)
        scene.notes = notes
        if PROBE and STATE["episodes"] == 0 and hasattr(module, "probe"):
            yield from module.probe(scene)
        yield from module.episode(scene)

    generator = commands()
    try:
        for joints in generator:
            if TASK_ENV.is_episode_end():
                break
            TASK_ENV.take_action(joints_to_action_dicts(np.asarray(joints, np.float32)[None])[0])
            steps += 1
            obs = TASK_ENV.get_obs()
            if recorder is not None:
                recorder.add(obs)
    except Exception:  # noqa: BLE001 - an expert bug fails this episode, not the client
        error = traceback.format_exc()
        print(f"[expert] episode raised:\n{error}", flush=True)
    finally:
        generator.close()
    if not TASK_ENV.is_episode_end():
        TASK_ENV.success[0] = False
        TASK_ENV.end_flag[0] = True
    success = bool(TASK_ENV.success[0])
    STATE["episodes"] += 1
    if recorder is not None:
        recorder.write(success, steps)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "variant": variant, "layout": layout, "success": success,
              "steps": steps, "seconds": round(time.time() - started, 1), "pid": os.getpid(),
              "notes": notes, "error": error.splitlines()[-1] if error else None}
    print(f"[expert] episode {json.dumps(record)}", flush=True)
    if LOG:
        with open(LOG, "a") as handle:
            handle.write(json.dumps(record) + "\n")
