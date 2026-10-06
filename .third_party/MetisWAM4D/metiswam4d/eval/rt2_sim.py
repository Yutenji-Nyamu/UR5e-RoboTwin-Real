"""RoboTwin 2.0 simulation side of the closed-loop evaluation.

Observation contract (identical to what ``RT2EpisodeDataset`` reads from the collected episodes):

- RGB: the collection writer stores ``cv2.imencode(".jpg", rgb)`` of the rendered array and training decodes
  the JPEG with PIL (``decode_jpeg``); live frames take exactly that path (``training_rgb``).
- Depth: the head camera depth in mm, stored as float16.
- Mask: role > 0 at the current frame = robot links (every articulation present before ``load_actors``) and the
  task entities (everything ``load_actors`` adds), from the per-pixel actor ids, as in ``rt2_track4d.replay_episode``.
  Training also marks clutter that moves later in the episode; that needs the future and is left out.
- Proprio: ``endpose`` -> EEF20 with the training formula (``read_eef20``).

Scenes: ``setup_demo(seed, is_test=True)`` with the official ``demo_clean`` / ``demo_randomized`` configs and step
limits.  A seed is admitted by the official expert check (``play_once`` succeeds); the same expert run provides the
scene description for the seen-language instruction.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys
import traceback

import cv2
import numpy as np
import yaml

from metiswam4d.data.rt2.eef import quat_xyzw_to_rot6d
from metiswam4d.data.rt2.episode_dataset import decode_jpeg

ROBOTWIN = Path("/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/third_party/RoboTwin")
CAMERAS = ("head_camera", "left_camera", "right_camera")
CLUTTER_LIMITS = dict(xlim=[-.59, .59], ylim=[-.34, .34], zlim=[.741])
SEEN_POOL = 16


def setup_runtime() -> None:
    """RoboTwin imports, renderer on this process' ``cuda:0``, clutter limits that do not drift across episodes."""
    for path in (ROBOTWIN, ROBOTWIN / "script", ROBOTWIN / "description" / "utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(ROBOTWIN)
    import sapien.core as sapien

    class DeviceEngine(sapien.Engine):
        def create_scene(self, config):
            sapien.physx.set_scene_config(config)
            return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device="cuda:0")])
    sapien.Engine = DeviceEngine

    # ``Base_Task.get_cluttered_table`` adds ``table_xy_bias`` into its mutable default ``xlim``/``ylim`` lists, so
    # every biased episode would shift the clutter region of all later episodes in the process.
    from envs._base_task import Base_Task
    original = Base_Task.get_cluttered_table
    if getattr(original, "_limits_isolated", False):
        return

    def get_cluttered_table(self, cluttered_numbers=10, xlim=None, ylim=None, zlim=None):
        return original(self, cluttered_numbers,
                        *(list(CLUTTER_LIMITS[name] if value is None else value)
                          for name, value in (("xlim", xlim), ("ylim", ylim), ("zlim", zlim))))
    get_cluttered_table._limits_isolated = True
    Base_Task.get_cluttered_table = get_cluttered_table


def task_args(task: str, task_config: str) -> dict:
    from envs import CONFIGS_PATH
    args = yaml.safe_load((ROBOTWIN / "task_config" / f"{task_config}.yml").read_text())
    args.update(task_name=task, task_config=task_config, policy_name="MetisWAM4D", eval_mode=True,
                eval_video_log=False, render_freq=0)
    kinds = yaml.safe_load(Path(CONFIGS_PATH, "_embodiment_config.yml").read_text())
    robot = kinds[args["embodiment"][0]]["file_path"]
    read = lambda root: yaml.safe_load(Path(root, "config.yml").read_text())
    args.update(dual_arm_embodied=True, left_robot_file=robot, right_robot_file=robot,
                left_embodiment_config=read(robot), right_embodiment_config=read(robot))
    cameras = yaml.safe_load(Path(CONFIGS_PATH, "_camera_config.yml").read_text())
    head = cameras[args["camera"]["head_camera_type"]]
    args.update(head_camera_h=head["h"], head_camera_w=head["w"])
    args["data_type"].update(rgb=True, depth=True, qpos=True, endpose=True, pointcloud=False,
                             mesh_segmentation=False, actor_segmentation=False)
    return args


_ROBOT = None      # one Robot (URDF + cuRobo planners) per process, reset into every new scene
_CLOSED = 0


def create_env(task: str, task_config: str, seed: int, episode_index: int):
    """``(env, foreground_ids)``; raises ``UnStableError`` for an unstable scene.

    ``Base_Task.load_robot`` resets an existing ``env.robot`` into the new scene instead of building the robot and
    its cuRobo planners again (the official collection / evaluation scripts reuse it across episodes the same way);
    the embodiment is the same for every task, so one robot serves all tasks of the process."""
    global _ROBOT
    env = getattr(importlib.import_module(f"envs.{task}"), task)()
    if _ROBOT is not None:
        env.robot = _ROBOT
    robot_ids: set[int] = set()
    task_ids: set[int] = set()
    load = env.load_actors

    def load_actors():
        before = {int(e.per_scene_id) for e in env.scene.entities}
        for articulation in env.scene.get_all_articulations():
            robot_ids.update(int(link.entity.per_scene_id) for link in articulation.get_links())
        load()
        task_ids.update(int(e.per_scene_id) for e in env.scene.entities if int(e.per_scene_id) not in before)
    env.load_actors = load_actors
    try:
        env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **task_args(task, task_config))
    except BaseException:
        close_env(env)
        raise
    finally:
        if getattr(env, "robot", None) is not None:
            _ROBOT = env.robot
    env.__dict__.pop("load_actors", None)
    env.test_num = episode_index
    return env, np.array(sorted(robot_ids | task_ids), dtype=np.int64)


def close_env(env) -> None:
    """Release a scene; the renderer asset cache is cleared every 5th scene (as ``collect_data.py``)."""
    global _CLOSED
    _CLOSED += 1
    try:
        env.close_env(clear_cache=_CLOSED % 5 == 0)
    except Exception:
        traceback.print_exc()


# Scene descriptions fixed by ``load_actors`` for experts whose ``play_once`` can raise before publishing them
# (put_bottles_dustbin: ``grasp_actor`` returns no actions when a grasp is unreachable -> IndexError).
SCENE_INFO = {
    "put_bottles_dustbin": lambda env: {"{A}": f"114_bottle/base{env.bottle_id[0]}", "{B}": f"114_bottle/base{env.bottle_id[1]}",
                                        "{C}": f"114_bottle/base{env.bottle_id[2]}", "{D}": "011_dustbin/base0"},
}


def expert_check(task: str, task_config: str, seed: int, episode_index: int) -> tuple[bool, dict | None, str]:
    """Official seed admission: the scripted expert must plan and succeed.  Returns (ok, scene info, note)."""
    env = None
    try:
        env, _ = create_env(task, task_config, seed, episode_index)
        info = env.play_once()
        ok = bool(env.plan_success and env.check_success())
        return ok, (info or {}).get("info"), "" if ok else "expert_failed"
    except Exception as exc:
        info = None
        if env is not None and task in SCENE_INFO:
            try:
                info = SCENE_INFO[task](env)
            except Exception:
                info = None
        return False, info, f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        if env is not None:
            close_env(env)


PLAN_METHODS = ("left_plan_path", "right_plan_path", "left_plan_multi_path", "right_plan_multi_path")


def scene_info_fast(task: str, task_config: str, seed: int, episode_index: int) -> dict | None:
    """Scene description of an already admitted seed without the expert's motion planning.

    ``play_once`` runs with every planner call answered by "stay at the current joints": the description it
    publishes depends only on the objects ``load_actors`` placed (verified per task against the real expert,
    ``FAST_INFO_TASKS``), while cuRobo planning is the dominant GPU cost of an expert run."""
    env, _ = create_env(task, task_config, seed, episode_index)
    robot = env.robot

    def hold(side: str) -> np.ndarray:
        return np.asarray(getattr(robot, f"get_{side}_arm_jointState")()[:-1], dtype=np.float64)[None]

    def single(side):
        return lambda *a, **k: {"status": "Success", "position": hold(side), "velocity": np.zeros_like(hold(side))}

    def multi(side):
        def plan(target_lst, *a, **k):
            n = len(target_lst)
            return {"status": ["Success"] * n, "position": [hold(side)] * n, "velocity": [np.zeros_like(hold(side))] * n}
        return plan

    for side in ("left", "right"):
        setattr(robot, f"{side}_plan_path", single(side))
        setattr(robot, f"{side}_plan_multi_path", multi(side))
    try:
        return (env.play_once() or {}).get("info")
    finally:
        for name in PLAN_METHODS:
            robot.__dict__.pop(name, None)
        close_env(env)


# Tasks whose stubbed-planner description equals the real expert's on every checked scene (2 per task, 98/98;
# scripts/eval/check_fast_scene_info.py, simeval_33760/parity/fast_scene_info.jsonl).
FAST_INFO_TASKS = frozenset((
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size", "click_alarmclock", "click_bell",
    "dump_bin_bigbin", "grab_roller", "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave", "pick_diverse_bottles",
    "pick_dual_bottles", "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch"))


def seen_instruction(task: str, info: dict, seed: int, kind: str = "seen", pool_size: int = SEEN_POOL) -> str:
    """Instruction of this scene from its ``kind`` language pool (the training prompts come from the seen pools).

    ``eval_policy.py`` draws from ``generate_episode_descriptions(task, [info], 100)[unseen]`` with the unseeded
    process-global ``random`` state; here the pool and the pick are functions of the seed."""
    from generate_episode_instructions import generate_episode_descriptions
    state = random.getstate()
    random.seed(seed)
    try:
        pool = generate_episode_descriptions(task, [info], pool_size)[0][kind]
    finally:
        random.setstate(state)
    digest = hashlib.sha256(f"instruction:{int(seed)}".encode()).digest()
    return str(pool[int.from_bytes(digest[:8], "big") % len(pool)])


OFFICIAL_POOL = 100      # eval_policy.py: generate_episode_descriptions(..., test_num)
INFRA_ERRORS = ("cannot create buffer", "CUDA", "out of memory", "kernel build failed")


class SeedRejected(RuntimeError):
    """The seed fails the official admission (``eval_policy.py`` moves on to the next seed)."""


def resolve_official(job: dict, folder: Path) -> dict:
    """Official RoboTwin protocol scene of ``job["environment_seed"]``: ``eval_policy.py`` runs the expert on every
    seed from ``100000 * (1 + seed)`` on and evaluates the seeds whose expert plans and succeeds; an unstable scene or
    any expert exception skips the seed.  Renderer / CUDA failures of this process are raised as errors (retried),
    not as rejections.  Cached in ``folder/scene.json`` (rejections too)."""
    path = folder / "scene.json"
    if path.exists():
        scene = json.loads(path.read_text())
    else:
        task, cfg, seed = job["task"], job["task_config"], int(job["environment_seed"])
        ok, info, note = expert_check(task, cfg, seed, int(job["protocol_episode_index"]))
        if not ok and any(marker in note for marker in INFRA_ERRORS):
            raise RuntimeError(f"expert check of seed {seed}: {note}")
        scene = {"environment_seed": seed, "info": info, "expert_ok": ok, "note": note}
        if ok:
            scene.update(instruction=seen_instruction(task, info, seed, job["instruction_type"], OFFICIAL_POOL),
                         instruction_type=job["instruction_type"])
        path.write_text(json.dumps(scene, indent=1, ensure_ascii=False, default=str))
    if not scene["expert_ok"]:
        raise SeedRejected(scene["note"])
    return scene


def resolve_scene(job: dict, folder: Path) -> dict:
    """Environment seed and instruction of one episode, cached in ``folder/scene.json``.

    Archived seeds (already admitted by the expert check) only need the scene description: ``scene_info_fast`` for
    the verified tasks, the expert otherwise; a seed that became unstable moves to the next seed.  Seed-search jobs
    scan their own disjoint block
    ``[seed_block, seed_block + block_size)`` in order and take the first admitted seed."""
    if job.get("admission") == "official":
        return resolve_official(job, folder)
    path = folder / "scene.json"
    if path.exists():
        return json.loads(path.read_text())
    task, cfg, ep = job["task"], job["task_config"], int(job["protocol_episode_index"])
    tried = []
    if job.get("environment_seed") is not None:
        candidates = [int(job["environment_seed"]) + k for k in range(10)]
        must_pass = False
    else:
        candidates = list(range(int(job["seed_block"]), int(job["seed_block"]) + int(job["block_size"])))
        must_pass = True
    scene = None
    for seed in candidates:
        info, ok, note = None, False, ""
        if not must_pass and task in FAST_INFO_TASKS:
            try:
                info = scene_info_fast(task, cfg, seed, ep)
                ok, note = True, "archived seed, description without expert planning"
            except Exception as exc:
                note = f"{type(exc).__name__}: {str(exc)[:200]}"
        if not info and "UnStableError" not in note:
            ok, info, note = expert_check(task, cfg, seed, ep)
        tried.append({"seed": seed, "ok": ok, "note": note})
        if ok or (not must_pass and "UnStableError" not in note):
            scene = {"environment_seed": seed, "info": info, "expert_ok": ok}
            break
    if scene is None:
        raise RuntimeError(f"no admissible seed for {job['key']}: {tried[-3:]}")
    if scene["info"] is not None:  # {} for tasks whose instructions have no placeholders (handover_block)
        scene.update(instruction=seen_instruction(task, scene["info"], scene["environment_seed"]),
                     instruction_type="seen")
    else:  # expert crashed before publishing the scene description: archived instruction
        scene.update(instruction=job.get("fallback_instruction"), instruction_type="unseen")
    if not scene["instruction"]:
        raise RuntimeError(f"no instruction for {job['key']}")
    scene["tried"] = tried
    path.write_text(json.dumps(scene, indent=1, ensure_ascii=False, default=str))
    return scene


def training_rgb(rgb: np.ndarray) -> np.ndarray:
    """Rendered uint8 RGB -> the pixels training sees: collection JPEG write + training JPEG decode."""
    ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(rgb, dtype=np.uint8))
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return np.asarray(decode_jpeg(encoded.tobytes()), dtype=np.uint8)


def eef20_from_obs(obs: dict) -> np.ndarray:
    arms = []
    for side in ("left", "right"):
        pose = np.asarray(obs["endpose"][f"{side}_endpose"], dtype=np.float32)
        grip = np.float32(obs["endpose"][f"{side}_gripper"])
        arms.append(np.concatenate((pose[:3], quat_xyzw_to_rot6d(pose[3:7]), [grip])))
    return np.concatenate(arms).astype(np.float32)


def observe(env, obs: dict, foreground_ids: np.ndarray) -> dict:
    """Training-domain policy inputs from a live observation."""
    cams = obs["observation"]
    seg = np.asarray(env.cameras.get_actor_seg_ids()["head_camera"]["actor_seg_ids"]).astype(np.int64)
    depth = np.asarray(cams["head_camera"]["depth"], dtype=np.float32).astype(np.float16).astype(np.float32)
    return {
        "rgb": {c: training_rgb(cams[c]["rgb"]) for c in CAMERAS},
        "depth_mm": depth,
        "mask": np.isin(seg, foreground_ids),
        "eef20": eef20_from_obs(obs),
    }


def policy_seed(environment_seed: int, episode: int, replan: int) -> int:
    digest = hashlib.sha256(f"metiswam4d-rt2:{environment_seed}:{episode}:{replan}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


__all__ = ["CAMERAS", "ROBOTWIN", "SeedRejected", "create_env", "eef20_from_obs", "expert_check", "observe",
           "policy_seed", "resolve_official", "resolve_scene", "seen_instruction", "setup_runtime", "task_args",
           "training_rgb"]
