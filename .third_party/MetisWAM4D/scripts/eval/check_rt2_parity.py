"""Training / evaluation input parity for the RoboTwin 2.0 closed loop.

Scene part: rebuild a collected episode's scene from its seed with the evaluation code (``rt2_sim.create_env``) and
compare the first live observation after the evaluation preprocessing with frame 0 as training reads it:
RGB x3 (stored JPEG decoded with PIL), head depth, foreground mask (``track4d.h5`` role > 0) and EEF20.

Model part (``--model``): the evaluation policy on training windows (inputs decoded from the episode files,
not from the simulator), normalised action MSE against the recorded actions over the 20 active dims.

    VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /usr/bin/python3.10 scripts/eval/check_rt2_parity.py --out <dir> [--model]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

RAW = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin")
DATA = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D")
DEFAULT_EPISODES = ("adjust_bottle/demo_clean_4d/0", "beat_block_hammer/demo_clean_4d/5",
                    "place_object_scale/demo_randomized_4d/7", "stack_blocks_two/demo_randomized_4d/11",
                    "open_laptop/demo_randomized_4d/3", "handover_block/demo_clean_4d/2")


def scene_parity(spec: str, out: Path) -> dict:
    from PIL import Image
    from metiswam4d.data.rt2.eef import read_eef20
    from metiswam4d.data.rt2.episode_dataset import decode_jpeg
    from metiswam4d.eval.rt2_sim import CAMERAS, create_env, observe
    task, variant, episode = spec.split("/")
    episode = int(episode)
    seed = int((RAW / task / variant / "seed.txt").read_text().split()[episode])
    cfg = "demo_randomized" if "randomized" in variant else "demo_clean"
    env, ids = create_env(task, cfg, seed, episode)
    try:
        obs = env.get_obs()
        live = observe(env, obs, ids)
        raw_rgb = {c: np.asarray(obs["observation"][c]["rgb"], dtype=np.uint8) for c in CAMERAS}
    finally:
        env.close_env(clear_cache=True)
    d = DATA / task / variant / f"episode{episode}"
    with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
        ref_rgb = {c: np.asarray(decode_jpeg(src[f"observation/{c}/rgb"][0]), dtype=np.uint8) for c in CAMERAS}
        ref_depth = src["observation/head_camera/depth"][0].astype(np.float32)
        ref_eef = read_eef20(src, np.array([0]))[0]
        ref_mask = t4d["role"][0] > 0
    mad = lambda a, b: float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    row = {"episode": spec, "seed": seed}
    for c in CAMERAS:
        row[f"{c}_rgb_mad"] = mad(live["rgb"][c], ref_rgb[c])
        # which channel order the stored frames have relative to the renderer (JPEG noise aside)
        row[f"{c}_raw_vs_train_mad"] = mad(raw_rgb[c], ref_rgb[c])
        row[f"{c}_raw_swapped_vs_train_mad"] = mad(raw_rgb[c][..., ::-1], ref_rgb[c])
    diff = np.abs(live["depth_mm"] - ref_depth)
    row["depth_p99_mm"] = float(np.percentile(diff, 99))
    row["depth_max_mm"] = float(diff.max())
    inter, union = (live["mask"] & ref_mask).sum(), (live["mask"] | ref_mask).sum()
    row["mask_iou"] = float(inter / max(union, 1))
    row["mask_live_only_px"] = int((live["mask"] & ~ref_mask).sum())
    row["mask_train_only_px"] = int((~live["mask"] & ref_mask).sum())
    row["eef_max_abs"] = float(np.abs(live["eef20"] - ref_eef).max())
    tag = spec.replace("/", "_")
    top = np.concatenate([live["rgb"]["head_camera"], ref_rgb["head_camera"],
                          np.abs(live["rgb"]["head_camera"].astype(int) - ref_rgb["head_camera"]).clip(0, 255).astype(np.uint8) * 4], 1)
    masks = np.concatenate([np.repeat(live["mask"][..., None], 3, -1), np.repeat(ref_mask[..., None], 3, -1),
                            np.repeat((live["mask"] ^ ref_mask)[..., None], 3, -1)], 1).astype(np.uint8) * 255
    Image.fromarray(np.concatenate([top, masks], 0)).save(out / f"scene_{tag}.png")
    return row


def model_parity(config: str, checkpoint: str, windows: int, out: Path) -> dict:
    from metiswam4d.data.rt2.eef import EEFNormalizer, read_eef20
    from metiswam4d.data.rt2.episode_dataset import decode_jpeg, read_index
    from metiswam4d.data.rt2.text_cache import load_instruction_index
    from metiswam4d.eval.rt2_policy import CAMERAS, RT2Policy
    policy = RT2Policy(config, checkpoint)
    rows = read_index(DATA, "index.jsonl", ("demo_clean_4d", "demo_randomized_4d"), None)
    prompts = load_instruction_index(DATA / "episode_instructions_sim_aligned.jsonl")
    rng = np.random.default_rng(0)
    eef = EEFNormalizer(policy.stage.data.rt2.alpha_stats)
    results = []
    for k in rng.choice(len(rows), windows, replace=False):
        row = rows[int(k)]
        d = DATA / row["task"] / row["variant"] / f"episode{row['episode']}"
        s = int(rng.integers(0, row["frames"] - 33))
        with h5py.File(d / "source.hdf5") as src, h5py.File(d / "track4d.h5") as t4d:
            request = {
                "rgb": {c: np.asarray(decode_jpeg(src[f"observation/{c}/rgb"][s]), dtype=np.uint8) for c in CAMERAS},
                "depth_mm": src["observation/head_camera/depth"][s].astype(np.float32),
                "mask": t4d["role"][s] > 0,
                "eef20": read_eef20(src, np.array([s]))[0],
            }
            gt = read_eef20(src, np.arange(s + 1, s + 33))
        key = f"{row['task']}/{row['variant']}/episode{row['episode']}"
        request.update(prompt=prompts[key][0], seed=int(k))
        pred = policy.predict(request)
        mse = float(np.square(eef.normalize(pred["eef20"]) - eef.normalize(gt)).mean())
        pos_cm = float(np.linalg.norm(pred["eef20"][:, [0, 1, 2]] - gt[:, [0, 1, 2]], axis=-1).mean() * 100)
        results.append({"window": f"{key}/s{s}", "action_mse_norm": mse, "left_pos_err_cm": pos_cm,
                        "rounds_run": pred["rounds_run"], "seconds": pred["inference_seconds"]})
        print(json.dumps(results[-1]), flush=True)
    summary = {"windows": len(results), "action_mse_norm": float(np.mean([r["action_mse_norm"] for r in results])),
               "left_pos_err_cm": float(np.mean([r["left_pos_err_cm"] for r in results])),
               "seconds_median": float(np.median([r["seconds"] for r in results[1:]] or [0])), "rows": results}
    (out / "model_parity.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", nargs="*", default=list(DEFAULT_EPISODES))
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--config", default="configs/rt2_direct_v3_coupled.yaml")
    ap.add_argument("--checkpoint", default="/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/"
                                            "rt2_direct_v3_coupled/checkpoints/save_step_0033760")
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.model:
        print(json.dumps({k: v for k, v in model_parity(args.config, args.checkpoint, args.windows, args.out).items()
                          if k != "rows"}), flush=True)
    if args.episodes:
        from metiswam4d.eval.rt2_sim import setup_runtime
        setup_runtime()
        rows = []
        for spec in args.episodes:
            try:
                rows.append(scene_parity(spec, args.out))
            except Exception as exc:
                rows.append({"episode": spec, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(rows[-1]), flush=True)
        (args.out / "scene_parity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
