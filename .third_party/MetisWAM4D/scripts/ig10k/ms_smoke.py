"""Smoke test for Imitator-Game ManiSkill tasks: reset + zero-action steps per L-level, dump camera frames.

Usage (after `source scripts/ig10k/ms_env.sh`):
    /usr/bin/python3.10 scripts/ig10k/ms_smoke.py --env TwoRobotPickAppleBasket-v1 --levels L0 L1 L2 L3
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

L_ENV_VARS = {"L1": "MANI_SKILL_L1", "L2": "MANI_SKILL_L2", "L3": "MANI_SKILL_L3"}


def set_level(level):
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

    for v in L_ENV_VARS.values():
        os.environ.pop(v, None)
    if level in L_ENV_VARS:
        os.environ[L_ENV_VARS[level]] = "1"
    L0_L3_utils.set_l1_enabled(level == "L1")
    L0_L3_utils.set_l2_enabled(level == "L2")
    L0_L3_utils.set_l3_enabled(level == "L3")
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def describe(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(describe(v, f"{prefix}{k}/"))
    else:
        a = to_np(tree)
        out[prefix.rstrip("/")] = [list(a.shape), str(a.dtype)]
    return out


def save_png(path, img):
    from PIL import Image

    img = to_np(img)
    if img.ndim == 4:
        img = img[0]
    if img.shape[-1] == 1:
        d = img[..., 0].astype(np.float32)
        valid = d > 0
        lo, hi = (np.percentile(d[valid], [1, 99]) if valid.any() else (0, 1))
        img = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img.astype(np.uint8)).save(path)


def hold_action(env):
    """Per-agent pd_joint_pos action that keeps the arms at their reset joint positions."""
    space = env.action_space
    agents = getattr(env.unwrapped.agent, "agents_dict", {})
    act = {}
    for k, sp in space.items():
        a = np.zeros(sp.shape, np.float32)
        if k in agents:
            q = to_np(agents[k].robot.get_qpos()).reshape(-1)
            n = sp.shape[-1] - 1
            a[..., :n] = q[:n]
            a[..., n] = sp.high.reshape(-1)[-1]
        act[k] = a
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="TwoRobotPickAppleBasket-v1")
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-backend", default="physx_cpu")
    ap.add_argument("--control-mode", default="pd_joint_pos")
    ap.add_argument("--out", default="/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/ig10k_sim_smoke")
    args = ap.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    out_root = Path(args.out) / args.env
    out_root.mkdir(parents=True, exist_ok=True)
    report = {}
    for level in args.levels:
        set_level(level)
        env_id = args.env.replace("-v1", "L3-v1") if level == "L3" and "L3" not in args.env else args.env
        t0 = time.time()
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="rgbd",
            control_mode=args.control_mode,
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            sensor_configs=dict(shader_pack="rt-fast"),
        )
        t_make = time.time() - t0
        obs, info = env.reset(seed=args.seed)
        t_reset = time.time() - t0 - t_make
        act = hold_action(env)
        qpos = None
        t1 = time.time()
        for _ in range(args.steps):
            obs, rew, term, trunc, info = env.step(act)
        t_step = (time.time() - t1) / args.steps

        lvl_dir = out_root / level
        lvl_dir.mkdir(exist_ok=True)
        for cam, data in obs.get("sensor_data", {}).items():
            if "rgb" in data:
                save_png(lvl_dir / f"{cam}_rgb.png", data["rgb"])
            if "depth" in data:
                save_png(lvl_dir / f"{cam}_depth.png", data["depth"])
        save_png(lvl_dir / "render.png", env.render())

        report[level] = dict(
            env_id=env_id,
            action_space=str(env.action_space),
            obs=describe(obs),
            info={k: to_np(v).tolist() for k, v in info.items() if np.asarray(to_np(v)).size <= 4},
            qpos_shape=None if qpos is None else list(qpos.shape),
            time_make_s=round(t_make, 2),
            time_reset_s=round(t_reset, 2),
            time_step_s=round(t_step, 4),
        )
        print(level, json.dumps({k: report[level][k] for k in ("env_id", "action_space", "info", "time_make_s", "time_reset_s", "time_step_s")}), flush=True)
        env.close()

    (out_root / "report.json").write_text(json.dumps(report, indent=1))
    print("wrote", out_root / "report.json")


if __name__ == "__main__":
    main()
