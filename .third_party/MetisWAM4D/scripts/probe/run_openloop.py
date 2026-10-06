#!/usr/bin/env python
"""Open-loop probe runner (P3.1 world replacement, P2.3 attention capture, P2.2 token subsets).

Per window all conditions share the same noise seed, so differences in the predicted action
block are caused by the intervention alone.  Results: one ``.npz`` per window under
``<out>/<pass>/<model>/`` with the normalised action block of every condition, native EEF
errors vs. ground truth and vs. the un-intervened run, attention aggregates and (for the
un-intervened run) the decoded predicted RGB / Track.

    source scripts/probe/env.sh
    CUDA_VISIBLE_DEVICES=0 $PY scripts/probe/run_openloop.py --model janus --pass p31 --shard 0/4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Intervention, JanusProbe, load_windows  # noqa: E402

OUT = Path(os.environ.get("PROBE_OUT", "/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes"))
SHIFT_FRAMES = 12
FEATURE_LAYERS = (5, 15, 25)


def window_seed(row: int, start: int) -> int:
    return (row * 1_000_003 + start * 7919) % (2**31 - 1)


class DonorCache:
    """Small LRU of captured K/V stores keyed by (row, start, static)."""

    def __init__(self, probe, capacity: int = 12):
        self.probe, self.capacity = probe, capacity
        self.store: OrderedDict = OrderedDict()

    def get(self, row: int, start: int, static: bool = False) -> dict:
        key = (row, start, static)
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        random.seed(window_seed(row, start))          # fixes the prompt choice inside read_window
        sample = self.probe.dataset.read_window(row, start)
        res = self.probe.predict_window(sample, window_seed(row, start), capture_kv=True, static_future=static)
        self.store[key] = res["kv"]
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return res["kv"]


def episode_frames(probe, row: int) -> int:
    import h5py
    with h5py.File(probe.dataset.rows[row]["source"], "r") as f:
        return int(f["observation/head_camera/depth"].shape[0])


def pick_donors(w: dict, windows: list[dict], rng: random.Random) -> dict:
    """Deterministic donors (one fixed event window per task) so the donor K/V cache is hit repeatedly."""
    tasks = sorted({x["task"] for x in windows})
    first_event = {}
    for x in windows:
        if x["kind"] == "event" and x["task"] not in first_event:
            first_event[x["task"]] = x
    same = [x for x in windows if x["task"] == w["task"] and x["key"] != w["key"] and x["kind"] == "event"]
    same_pick = first_event.get(w["task"])
    if same_pick is None or same_pick["key"] == w["key"]:
        same_pick = same[0] if same else None
    other_task = tasks[(tasks.index(w["task"]) + len(tasks) // 2) % len(tasks)]
    return dict(same_task=same_pick, other_task=first_event.get(other_task))


def run_p31(probe, w: dict, windows: list[dict], cache: DonorCache, rng: random.Random) -> dict:
    row, start = w["row"], w["start"]
    seed = window_seed(row, start)
    random.seed(seed)                                 # fixes the prompt choice inside read_window
    sample = probe.dataset.read_window(row, start)
    gt = sample["action"].numpy()
    out: dict = dict(meta=json.dumps(w), seed=seed, gt_action=gt, prompt=sample["prompt"], bucket=sample["bucket"])
    actions: dict[str, np.ndarray] = {}

    full = probe.predict_window(sample, seed, capture_attn=True, decode=True)
    actions["full"] = full["action_norm"]
    seg = full["segments"]
    out["segments"] = json.dumps(dict(ranges=seg.ranges, video_grid=seg.video_grid, track_grid=seg.track_grid))
    A = full["attn_by_head"]                      # [S, L, h, K]
    out["attn_step_layer"] = A.sum(axis=2).astype(np.float32)          # [S, L, K]  summed over heads & queries
    keep_steps = [0, A.shape[0] // 2, A.shape[0] - 1]
    out["attn_steps_kept"] = np.asarray(keep_steps)
    out["attn_by_head"] = A[keep_steps].astype(np.float16)              # [3, L, h, K]
    out["attn_by_query"] = full["attn_by_query"][keep_steps].astype(np.float16)   # [3, L, A, K]
    out["pred_rgb"] = full["pred_rgb"]
    out["pred_track"] = full["pred_track"]

    for name, iv in [("drop_video", Intervention("drop", ("video_future",))),
                     ("drop_track", Intervention("drop", ("track_future",))),
                     ("drop_world", Intervention("drop")),
                     ("zero_video", Intervention("zero", ("video_future",))),
                     ("pool_video", Intervention("pool", ("video_future",))),
                     ("pool_world", Intervention("pool"))]:
        actions[name] = probe.predict_window(sample, seed, intervention=iv)["action_norm"]

    donors = pick_donors(w, windows, rng)
    T = episode_frames(probe, row)
    shift = start + SHIFT_FRAMES if start + SHIFT_FRAMES + 32 < T else start - SHIFT_FRAMES
    donor_specs = {
        "swap_same_task": (donors["same_task"]["row"], donors["same_task"]["start"], False) if donors["same_task"] else None,
        "swap_other_task": (donors["other_task"]["row"], donors["other_task"]["start"], False) if donors["other_task"] else None,
        "swap_shift": (row, shift, False) if 0 < shift and shift + 32 < T else None,
        "swap_static": (row, start, True),
    }
    out["donors"] = json.dumps({k: (list(v) if v else None) for k, v in donor_specs.items()})
    for name, spec in donor_specs.items():
        if spec is None:
            continue
        kv = cache.get(*spec)
        for tgt, suffix in ((("video_future",), "_video"), (("video_future", "track_future"), "_world")):
            actions[name + suffix] = probe.predict_window(sample, seed, intervention=Intervention("replace", tgt, donor=kv))["action_norm"]
    errors = {}
    for name, a in actions.items():
        errors[name] = dict(vs_gt=probe.action_errors(a, gt), vs_full=probe.action_errors(a, actions["full"]))
        out[f"action_{name}"] = a
    out["errors"] = json.dumps(errors)
    return out


def run_p31c(probe, w: dict, windows: list[dict], cache: DonorCache, rng: random.Random) -> dict:
    """Current-frame variant of P3.1: intervene on the CLEAN (observed) video keys the action reads.

    Used for FastWAM, whose action branch never sees future video, and as a reference for the others
    (how much of the action is driven by the current visual observation vs. proprio + text).
    """
    row, start = w["row"], w["start"]
    seed = window_seed(row, start)
    random.seed(seed)
    sample = probe.dataset.read_window(row, start)
    gt = sample["action"].numpy()
    out: dict = dict(meta=json.dumps(w), seed=seed, gt_action=gt, prompt=sample["prompt"], bucket=sample["bucket"])
    actions: dict[str, np.ndarray] = {}
    full = probe.predict_window(sample, seed, capture_attn=True, decode=False)
    actions["full"] = full["action_norm"]
    seg = full["segments"]
    out["segments"] = json.dumps(dict(ranges=seg.ranges, video_grid=seg.video_grid, track_grid=seg.track_grid))
    A = full["attn_by_head"]
    out["attn_step_layer"] = A.sum(axis=2).astype(np.float32)
    keep_steps = [0, A.shape[0] // 2, A.shape[0] - 1]
    out["attn_steps_kept"] = np.asarray(keep_steps)
    out["attn_by_head"] = A[keep_steps].astype(np.float16)
    out["attn_by_query"] = full["attn_by_query"][keep_steps].astype(np.float16)
    tgt = ("video_clean",)
    for name, iv in [("drop_clean", Intervention("drop", tgt)), ("zero_clean", Intervention("zero", tgt)), ("pool_clean", Intervention("pool", tgt))]:
        actions[name] = probe.predict_window(sample, seed, intervention=iv)["action_norm"]
    donors = pick_donors(w, windows, rng)
    for name, d in (("swap_clean_same_task", donors["same_task"]), ("swap_clean_other_task", donors["other_task"])):
        if d is None:
            continue
        kv = cache.get(d["row"], d["start"], False)
        actions[name] = probe.predict_window(sample, seed, intervention=Intervention("replace", tgt, donor=kv))["action_norm"]
    errors = {}
    for name, a in actions.items():
        errors[name] = dict(vs_gt=probe.action_errors(a, gt), vs_full=probe.action_errors(a, actions["full"]))
        out[f"action_{name}"] = a
    out["errors"] = json.dumps(errors)
    return out


def subset_masks(gt: dict, seg, rng: np.random.Generator, fractions=(0.2, 0.05, 0.01)) -> dict[str, dict[str, torch.Tensor]]:
    """P2.2 keep-masks over the future world tokens.

    Video future tokens: ``[F-1, 12, 10]`` (rows 0-7 = head view); Track future tokens: ``[F-1, 8, 10]``.
    Structured subsets are chosen among head-view tokens by a GT score (interaction fraction, |dc|, motion
    magnitude, object fraction); ``random_*`` samples uniformly among head tokens; ``head_only`` / ``wrist_only`` /
    ``frame1_only`` / ``frame2_only`` are the view / time references.  The same rule is applied to Track tokens.
    """
    fv, vh, vw = seg.video_grid
    ft, th, tw = seg.track_grid
    n_vf, n_tf = (fv - 1) * vh * vw, max(ft - 1, 0) * th * tw
    out: dict[str, dict[str, torch.Tensor]] = {}

    def video_head_index():
        idx = np.arange(n_vf).reshape(fv - 1, vh, vw)
        return idx[:, :8].reshape(-1), idx[:, 8:].reshape(-1)

    head_idx, wrist_idx = video_head_index()
    scores = {
        "inter": (gt["video"]["inter"], gt["track"]["inter"]),
        "dc": (gt["dc"], gt["dc"]),
        "mag": (gt["mag"], gt["mag"]),
        "obj": (gt["video"]["obj"], gt["track"]["obj"]),
    }

    def make(video_keep: np.ndarray, track_keep: np.ndarray | None):
        d = {"video_future": torch.from_numpy(video_keep.astype(bool))}
        if n_tf:
            d["track_future"] = torch.from_numpy(track_keep.astype(bool))
        return d

    def topk(score2, k):
        flat = score2.reshape(-1) + 1e-6 * rng.random(score2.size)   # random tie-break
        keep = np.zeros(flat.size, dtype=bool)
        keep[np.argsort(-flat)[:k]] = True
        return keep

    v_all = np.zeros(n_vf, dtype=bool)
    v_head = v_all.copy(); v_head[head_idx] = True
    v_wrist = v_all.copy(); v_wrist[wrist_idx] = True
    t_all = np.ones(n_tf, dtype=bool)
    out["head_only"] = make(v_head, t_all)
    out["wrist_only"] = make(v_wrist, np.zeros(n_tf, dtype=bool))
    for f in range(fv - 1):
        vk = np.zeros(n_vf, dtype=bool); vk[f * vh * vw:(f + 1) * vh * vw] = True
        tk = np.zeros(n_tf, dtype=bool)
        if n_tf:
            tk[f * th * tw:(f + 1) * th * tw] = True
        out[f"frame{f + 1}_only"] = make(vk, tk)
    n_head, n_track = len(head_idx), n_tf
    for frac in fractions:
        kv, kt = max(1, round(frac * n_head)), max(1, round(frac * max(n_track, 1)))
        vk = np.zeros(n_vf, dtype=bool); vk[rng.choice(head_idx, kv, replace=False)] = True
        tk = np.zeros(n_tf, dtype=bool)
        if n_tf:
            tk[rng.choice(n_tf, kt, replace=False)] = True
        out[f"random_{int(frac * 100)}"] = make(vk, tk)
        for nm, (sv, st) in scores.items():
            vk = np.zeros(n_vf, dtype=bool)
            vk[head_idx[topk(sv, kv)]] = True                      # sv: [2, 8, 10] head grid, same order as head_idx
            tk = topk(st, kt) if n_tf else np.zeros(0, dtype=bool)
            out[f"{nm}_{int(frac * 100)}"] = make(vk, tk)
    return out


def run_p22(probe, w: dict, rng: np.random.Generator) -> dict:
    from analyze_openloop import window_gt
    row, start = w["row"], w["start"]
    seed = window_seed(row, start)
    random.seed(seed)
    sample = probe.dataset.read_window(row, start)
    gt_action = sample["action"].numpy()
    full = probe.predict_window(sample, seed, capture_kv=True)
    seg = full["segments"]
    gt = window_gt(w["key"], start)
    masks = subset_masks(gt, seg, rng)
    # P1.1 features: value projections of the future world tokens at three depths, last denoising step
    feats = {}
    if "feats" in full["kv"]:                       # FlowWAM: captured per-layer features [1, 720, 3072]
        for layer in FEATURE_LAYERS:
            f = full["kv"]["feats"][layer][0].float().numpy().astype(np.float16)
            r = seg.ranges
            feats[f"feat_video_future_L{layer}"] = f[r["video_future"][0]:r["video_future"][1]]
            feats[f"feat_flow_future_L{layer}"] = f[r["flow_future"][0]:r["flow_future"][1]]
    else:
        last_step = max(s for s, _ in full["kv"])
        for layer in FEATURE_LAYERS:
            entry = full["kv"].get((last_step, layer))
            if entry is None:
                continue
            for segname, (k, v) in entry.items():
                if v.shape[1]:
                    feats[f"feat_{segname}_L{layer}"] = v[0].float().numpy().astype(np.float16)
    gt_maps = {f"gt_{k}": np.asarray(v, dtype=np.float32) for k, v in gt.items() if not isinstance(v, dict)}
    gt_maps.update({f"gt_video_{k}": v for k, v in gt["video"].items()})
    gt_maps.update({f"gt_track_{k}": v for k, v in gt["track"].items()})
    actions = {"full": full["action_norm"], "drop_world": probe.predict_window(sample, seed, intervention=Intervention("drop"))["action_norm"]}
    for name, keep in masks.items():
        targets = tuple(keep.keys())
        actions[name] = probe.predict_window(sample, seed, intervention=Intervention("keep", targets, keep=keep))["action_norm"]
    out: dict = dict(meta=json.dumps(w), seed=seed, gt_action=gt_action,
                     segments=json.dumps(dict(ranges=seg.ranges, video_grid=seg.video_grid, track_grid=seg.track_grid)))
    errors = {}
    for name, a in actions.items():
        errors[name] = dict(vs_gt=probe.action_errors(a, gt_action), vs_full=probe.action_errors(a, actions["full"]))
        out[f"action_{name}"] = a
    out["errors"] = json.dumps(errors)
    out["keep_counts"] = json.dumps({k: {s: int(m.sum()) for s, m in v.items()} for k, v in masks.items()})
    out.update(feats)
    out.update(gt_maps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="janus", choices=["janus", "alpha", "xwam", "flowwam", "effwam", "fastwam"])
    ap.add_argument("--pass", dest="which", default="p31", choices=["p31", "p31c", "p22"])
    ap.add_argument("--windows", default=str(OUT / "windows.jsonl"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    windows = load_windows(a.windows)
    mine = windows[i::n]
    if a.limit:
        mine = mine[: a.limit]
    out_dir = Path(a.out) / a.which / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.model == "janus":
        probe = JanusProbe()
    elif a.model == "alpha":
        from harness_alpha import AlphaProbe
        probe = AlphaProbe()
    elif a.model == "xwam":
        from harness_xwam import XWAMProbe
        probe = XWAMProbe()
    elif a.model == "flowwam":
        from harness_flowwam import FlowWAMProbe
        probe = FlowWAMProbe()
    elif a.model == "effwam":
        from harness_effwam import EffWAMProbe
        probe = EffWAMProbe()
    elif a.model == "fastwam":
        from harness_fastwam import FastWAMProbe
        probe = FastWAMProbe()
    else:
        raise NotImplementedError(a.model)
    cache = DonorCache(probe)
    rng = random.Random(1234 + i)
    nrng = np.random.default_rng(1234 + i)
    log = open(out_dir / f"shard_{i}_of_{n}.log", "a")
    t0 = time.time()
    for j, w in enumerate(mine):
        name = f"{w['key'].replace('/', '__')}__f{w['start']}.npz"
        path = out_dir / name
        if path.exists():
            continue
        t = time.time()
        try:
            res = (run_p31(probe, w, windows, cache, rng) if a.which == "p31" else
                   run_p31c(probe, w, windows, cache, rng) if a.which == "p31c" else run_p22(probe, w, nrng))
            np.savez_compressed(path, **res)
            e = json.loads(res["errors"])
            if a.which == "p31c":
                pick = ["drop_clean", "zero_clean", "pool_clean", "swap_clean_same_task", "swap_clean_other_task"]
                key = "pos_cm" if np.isfinite(e["full"]["vs_gt"]["pos_cm"]) else "joint_deg"
                msg = (f"[{j + 1}/{len(mine)}] {w['key']} f{w['start']} {w['kind']} {time.time() - t:.0f}s full {e['full']['vs_gt'][key]:.2f} | "
                       + " ".join(f"{k} {e[k]['vs_full'][key]:.2f}" for k in pick if k in e) + f" | {(time.time() - t0) / 60:.0f} min")
            elif a.which == "p31":
                msg = (f"[{j + 1}/{len(mine)}] {w['key']} f{w['start']} {w['kind']} {time.time() - t:.0f}s "
                       f"full {e['full']['vs_gt']['pos_cm']:.2f}cm drop_video {e['drop_video']['vs_full']['pos_cm']:.1f} "
                       f"drop_track {e['drop_track']['vs_full']['pos_cm']:.2f} "
                       f"swap_same {e.get('swap_same_task_video', {}).get('vs_full', {}).get('pos_cm', float('nan')):.2f} "
                       f"swap_other {e.get('swap_other_task_video', {}).get('vs_full', {}).get('pos_cm', float('nan')):.2f} "
                       f"static {e['swap_static_video']['vs_full']['pos_cm']:.2f} | {(time.time() - t0) / 60:.0f} min")
            else:
                pick = ["drop_world", "head_only", "wrist_only", "random_5", "inter_5", "dc_5", "mag_5", "random_20", "inter_20"]
                msg = (f"[{j + 1}/{len(mine)}] {w['key']} f{w['start']} {w['kind']} {time.time() - t:.0f}s full {e['full']['vs_gt']['pos_cm']:.2f}cm | "
                       + " ".join(f"{k} {e[k]['vs_full']['pos_cm']:.1f}" for k in pick if k in e) + f" | {(time.time() - t0) / 60:.0f} min")
        except Exception:
            msg = f"[{j + 1}/{len(mine)}] {w['key']} f{w['start']} ERROR\n{traceback.format_exc()[-1500:]}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()


if __name__ == "__main__":
    main()
