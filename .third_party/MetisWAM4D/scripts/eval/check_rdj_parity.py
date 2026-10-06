"""Training <-> evaluation parity for the RoboDojo policy (``metiswam4d/eval/rdj_*``).

    PYTHONPATH=.:/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824 CUDA_VISIBLE_DEVICES=0 \
        /usr/bin/python3.10 scripts/eval/check_rdj_parity.py --config configs/stage3_robodojo_v2_coupled.yaml \
        --model <eval dir>/checkpoint/model_bf16_stepN.pt --out <eval dir>/parity [--windows 8]

1. geometry: the SAPIEN robot renderer fed with the stored joint states / camera reproduces the training head depth
   and mask of random frames of several episodes (IoU, mm difference);
2. end effector: ``eef20 -> take_action dicts -> eef20`` round trip and agreement with the official ``state/*_ee_poses``;
3. model: fixed held-out windows through ``RDJPolicy.predict_batch`` (inputs rebuilt from the per-camera JPEGs, the
   stored robot depth / role and the stored eef20), against the training action targets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json")

from metiswam4d.config import load_stage_config  # noqa: E402
from metiswam4d.data.robodojo.episode_dataset import RDJEpisodeDataset, decode_bgr_jpeg, read_index  # noqa: E402
from metiswam4d.data.rt2.eef import gather_unified  # noqa: E402
from metiswam4d.data.rt2.episode_dataset import ACTION_STEPS  # noqa: E402
from metiswam4d.eval.rdj_observation import (  # noqa: E402
    CAMERAS, RobotGeometry, eef20_from_observation, eef20_to_action_dicts, pose_wxyz_to_arm10)


def geometry_check(root: Path, rows: list[dict], frames_per_episode: int, rng: random.Random) -> dict:
    geometry = RobotGeometry()
    ious, depth_max, depth_mean = [], [], []
    for row in rows:
        d = root / row["task"] / row["variant"] / f"episode{row['episode']}"
        with h5py.File(d / "source.hdf5") as src:
            n = src["joint_state/vector"].shape[0]
            for t in sorted(rng.sample(range(n), frames_per_episode)):
                k = src["observation/head_camera/intrinsic_cv"][t]
                obs = {"vision": {"cam_head": {"intrinsic_matrix": k, "extrinsic_matrix": src["observation/head_camera/cam2world_gl"][t],
                                               "color": np.zeros((240, 320, 3), np.uint8)}},
                       "state": {}}
                q = src["joint_state/vector"][t]
                obs["state"] = {"left_arm_joint_state": q[:6], "left_ee_joint_state": [float(q[6])],
                                "right_arm_joint_state": q[7:13], "right_ee_joint_state": [float(q[13])]}
                depth_mm, mask = geometry(obs)
                ref_depth = src["observation/head_camera/depth"][t].astype(np.float32)
                ref_mask = src["observation/head_camera/instance_id"][t] > 0
                union = (mask | ref_mask).sum()
                ious.append(float((mask & ref_mask).sum() / union) if union else 1.0)
                both = mask & ref_mask
                diff = np.abs(depth_mm - ref_depth)[both]
                depth_max.append(float(diff.max()) if diff.size else 0.0)
                depth_mean.append(float(diff.mean()) if diff.size else 0.0)
    return {"frames": len(ious), "mask_iou_min": min(ious), "mask_iou_mean": float(np.mean(ious)),
            "depth_abs_mm_max": max(depth_max), "depth_abs_mm_mean": float(np.mean(depth_mean))}


def eef_check(root: Path, rows: list[dict]) -> dict:
    round_trip, quat_err, state_err = [], [], []
    for row in rows:
        d = root / row["task"] / row["variant"] / f"episode{row['episode']}"
        with h5py.File(d / "track4d.h5") as t4d, h5py.File(d / "raw.hdf5") as raw:
            eef = t4d["eef20"][:200]
            actions = eef20_to_action_dicts(eef)
            for i, action in enumerate(actions):
                back = np.concatenate([pose_wxyz_to_arm10(action[f"{s}_ee_pose"], action[f"{s}_ee_joint_state"][0])
                                       for s in ("left", "right")])
                round_trip.append(float(np.abs(back - eef[i]).max()))
                for s, start in (("left", 0), ("right", 10)):
                    official = raw[f"state/{s}_ee_poses"][i]
                    q_pred, q_ref = action[f"{s}_ee_pose"][3:], official[3:7]
                    quat_err.append(float(min(np.abs(q_pred - q_ref).max(), np.abs(q_pred + q_ref).max())))
                    state_err.append(float(np.abs(action[f"{s}_ee_pose"][:3] - official[:3]).max()))
            obs = {"state": {"left_ee_pose": raw["state/left_ee_poses"][0], "right_ee_pose": raw["state/right_ee_poses"][0],
                             "left_ee_joint_state": [float(raw["state/left_ee_joint_states"][0, 0])],
                             "right_ee_joint_state": [float(raw["state/right_ee_joint_states"][0, 0])]}}
            state_err.append(float(np.abs(eef20_from_observation(obs) - eef[0]).max()))
    return {"round_trip_max": max(round_trip), "quat_vs_official_max": max(quat_err), "pose_vs_official_max": max(state_err)}


def model_check(config: str, model_file: str, root: Path, windows: int, out: Path) -> dict:
    import torch
    from metiswam4d.eval.rdj_policy import RDJPolicy
    import dataclasses
    stage = load_stage_config(config)
    dataset = RDJEpisodeDataset(dataclasses.replace(stage.data.robodojo, split="val", fixed_windows=True,
                                                    samples_per_episode=1))
    policy = RDJPolicy(config, model_file)
    picks = np.linspace(0, len(dataset) - 1, windows).astype(int)
    requests, targets, keys = [], [], []
    for index in picks:
        item = dataset[int(index)]
        key = item["key"]
        task, variant, episode, s = key.split("/")
        s = int(s[1:])
        d = root / task / variant / episode
        with h5py.File(d / "source.hdf5") as src:
            rgb = {c: np.asarray(decode_bgr_jpeg(src[f"observation/{c}/rgb"][s]), dtype=np.uint8) for c in CAMERAS}
        with h5py.File(d / "track4d.h5") as t4d:
            eef = t4d["eef20"][s]
        requests.append({"rgb": rgb, "depth_mm": item["head_depth_mm"].numpy(), "mask": item["head_mask"].numpy(),
                         "eef20": eef, "prompt": item["prompt"], "seed": 1000 + int(index)})
        targets.append(item["action"].numpy())
        keys.append(key)
    outs = policy.predict_batch(requests)
    rows = []
    for key, out_i, target in zip(keys, outs, targets):
        pred_norm = policy.eef.normalize(out_i["eef20_base"])
        gt = gather_unified(target)
        gt_base = policy.eef.denormalize(gt)
        pos = np.abs(out_i["eef20_base"][:, [0, 1, 2, 10, 11, 12]] - gt_base[:, [0, 1, 2, 10, 11, 12]])
        rows.append({"key": key, "norm_mse": float(((pred_norm - gt) ** 2).mean()),
                     "pos_err_cm_mean": float(pos.mean() * 100), "pos_err_cm_max": float(pos.max() * 100),
                     "gripper_err": float(np.abs(out_i["eef20_base"][:, [9, 19]] - gt_base[:, [9, 19]]).mean()),
                     "hold_pos_err_cm": float(np.abs(gt_base[:, [0, 1, 2, 10, 11, 12]] - gt_base[:1, [0, 1, 2, 10, 11, 12]]).mean() * 100),
                     "rounds_run": out_i["rounds_run"], "seconds": out_i["inference_seconds"]})
    return {"windows": rows, "norm_mse_mean": float(np.mean([r["norm_mse"] for r in rows])),
            "pos_err_cm_mean": float(np.mean([r["pos_err_cm_mean"] for r in rows])),
            "hold_baseline_cm_mean": float(np.mean([r["hold_pos_err_cm"] for r in rows]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--skip-geometry", action="store_true")
    args = ap.parse_args()
    stage = load_stage_config(args.config)
    root = Path(stage.data.robodojo.root)
    rng = random.Random(0)
    rows = read_index(root, stage.data.robodojo.index_file, "val", None)
    rows = rng.sample(rows, args.episodes)
    report = {"episodes": [f"{r['task']}/episode{r['episode']}" for r in rows]}
    if not args.skip_geometry:
        report["geometry"] = geometry_check(root, rows, args.frames, rng)
        print(json.dumps(report["geometry"]), flush=True)
    report["eef"] = eef_check(root, rows)
    print(json.dumps(report["eef"]), flush=True)
    if args.model:
        report["model"] = model_check(args.config, args.model, root, args.windows, args.out)
        print(json.dumps({k: v for k, v in report["model"].items() if k != "windows"}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rdj_parity.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
