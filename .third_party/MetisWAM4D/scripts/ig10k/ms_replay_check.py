"""Replay IG-10K sim episodes in the Imitator-Game ManiSkill env and check success.

The dataset stores no seed; the collector tries seeds 0, 1, 2, ... and skips failed plans, so episode i used
some seed >= i. The seed is recovered by matching the first zed2i frame against env resets.

Usage (after `source scripts/ig10k/ms_env.sh`):
    /usr/bin/python3.10 scripts/ig10k/ms_replay_check.py --env-dir L0_TwoRobotPickAppleBasket-v1 --episodes 0 1 2
"""
import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

SIM_ROOT = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/IG-10K-Dataset/imitator_sim_v1_zed2i")
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


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_episodes(env_dir):
    data = pq.read_table(sorted((env_dir / "data").glob("chunk-*/*.parquet"))[0]).to_pandas()
    meta = pq.read_table(sorted((env_dir / "meta" / "episodes").glob("chunk-*/*.parquet"))[0]).to_pandas()
    return data, meta


def first_frame(env_dir, meta_row, fps=30):
    path = env_dir / "videos" / "observation.images.zed2i" / "chunk-000" / "file-000.mp4"
    target = int(round(float(meta_row["videos/observation.images.zed2i/from_timestamp"]) * fps))
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == target:
                return f.to_ndarray(format="rgb24")
    raise IndexError(target)


def split_action(a16, space, swap=False):
    keys = list(space.keys())
    a, b = a16[:8].astype(np.float32), a16[8:16].astype(np.float32)
    return {keys[0]: b, keys[1]: a} if swap else {keys[0]: a, keys[1]: b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", default="L0_TwoRobotPickAppleBasket-v1")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed-window", type=int, default=40)
    ap.add_argument("--sim-backend", default="physx_cpu")
    ap.add_argument("--mirror-robot-pose", type=int, choices=[0, 1], default=0,
                    help="L2 LR mirror of robot root poses; the dataset and the official eval use 0 (library default is 1)")
    ap.add_argument("--swap-arms", action="store_true", help="feed dataset arm-0 actions to robot 1 and vice versa")
    ap.add_argument("--out", default="/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/ig10k_sim_smoke/replay")
    args = ap.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from PIL import Image

    level, env_name = args.env_dir.split("_", 1)
    set_level(level)
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

    L0_L3_utils.set_lr_mirror_robot_pose_enabled(bool(args.mirror_robot_pose))
    env_id = env_name.replace("-v1", "L3-v1") if level == "L3" else env_name
    env_dir = SIM_ROOT / args.env_dir
    data, meta = load_episodes(env_dir)
    tag = "" if args.mirror_robot_pose == 0 else f"_pose{args.mirror_robot_pose}"
    out = Path(args.out) / (args.env_dir + tag + ("_swap" if args.swap_arms else ""))
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        env_id, num_envs=1, obs_mode="rgbd", control_mode="pd_joint_pos", render_mode="rgb_array",
        sim_backend=args.sim_backend, sensor_configs=dict(shader_pack="rt-fast"), max_episode_steps=2000,
    )
    results = []
    for ep in args.episodes:
        row = meta[meta["episode_index"] == ep].iloc[0]
        frames = data[data["episode_index"] == ep].sort_values("frame_index")
        actions = np.stack(frames["action.qpos_gripper_actions"].to_numpy())
        qpos0 = np.asarray(frames["observation.qpos_gripper_states"].iloc[0])
        ref = first_frame(env_dir, row).astype(np.float32)

        best = (None, np.inf)
        for s in range(ep, ep + args.seed_window):
            obs, _ = env.reset(seed=s)
            img = to_np(obs["sensor_data"]["zed2i"]["rgb"])[0].astype(np.float32)
            mse = float(((img - ref) ** 2).mean())
            if mse < best[1]:
                best = (s, mse)
        seed, mse = best
        obs, _ = env.reset(seed=seed)
        img0 = to_np(obs["sensor_data"]["zed2i"]["rgb"])[0]
        succ_any, succ_last = False, False
        for a in actions:
            obs, rew, term, trunc, info = env.step(split_action(a, env.action_space, args.swap_arms))
            succ_last = bool(to_np(info["success"]).reshape(-1)[0])
            succ_any |= succ_last
        last = to_np(obs["sensor_data"]["zed2i"]["rgb"])[0]
        Image.fromarray(np.concatenate([ref.astype(np.uint8), img0, last], 1)).save(out / f"ep{ep:03d}_ref_reset_last.png")
        final_info = {k: to_np(v).reshape(-1)[0].item() for k, v in info.items()
                      if to_np(v).size == 1 and k != "elapsed_steps"}
        r = dict(episode=ep, seed=seed, first_frame_mse=round(mse, 2), steps=len(actions), success_any=succ_any,
                 success_last=succ_last, qpos0_dim=int(qpos0.shape[0]), final_info=final_info)
        results.append(r)
        print(json.dumps(r), flush=True)
    env.close()
    (out / "replay.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
