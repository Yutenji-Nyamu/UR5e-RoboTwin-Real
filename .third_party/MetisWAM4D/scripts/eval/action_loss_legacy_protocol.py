"""Action loss of a checkpoint under the JanusAct4D (legacy) training protocol, for a like-for-like comparison.

Legacy protocol (``janusact4d_*/objective.py``): sigma_action = shifted(U(0, 1)) (Alpha's uniform timesteps, shift 5),
Track noised with the *same* sigma as Action, Video with an independent shifted U(0, 1); no shortcut dropout;
velocity MSE over the 20 active EEF dims and 32 steps; per-sigma bins by thirds of the shifted sigma.  The reference
numbers are JanusAct4D rt2_v1 (19k-30k steps) 0.00045-0.00052 (bins low 0.00106 / mid 0.00069 / high 0.00030) and
rdj_v1 (30k-40k) 0.0005-0.0006.

    CUDA_VISIBLE_DEVICES=1 /usr/bin/python3.10 scripts/eval/action_loss_legacy_protocol.py \
        --config configs/rt2_direct_v3_coupled.yaml --checkpoint <ckpt dir> [--windows 256] [--batch 4] [--split val]

Windows are drawn with a fixed seed from the training split (the legacy numbers are training losses); ``--split val``
uses the RoboDojo held-out split.  Nothing is written to the run directory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from metiswam4d.build import build_model
from metiswam4d.config import load_stage_config
from metiswam4d.objectives import action_loss, prepare_training_step
from metiswam4d.schedule import AsyncSchedule, NoiseAssignment, NoiseMixture, flow_shift
from metiswam4d.train.checkpoint import load_model_weights
from metiswam4d.train.train import move_batch


class LegacyMixture(NoiseMixture):
    """sigma_A = sigma_T = shifted(U), sigma_V = shifted(U') independent."""

    def sample(self, batch_size, schedule: AsyncSchedule, *, generator=None, device="cpu") -> NoiseAssignment:
        u = torch.rand(batch_size, generator=generator, dtype=torch.float64).clamp(min=1e-3)
        v = torch.rand(batch_size, generator=generator, dtype=torch.float64).clamp(min=1e-3)
        body = flow_shift(u, 5.0).float().to(device)
        sigma = {}
        for name in schedule.modalities:
            sigma[name] = flow_shift(v, 5.0).float().to(device) if name == "video" else body.clone()
        return NoiseAssignment(sigma, torch.zeros(batch_size, dtype=torch.long, device=device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="checkpoints/step_XXXXXXX (DCP) directory")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--split", default=None, help="RoboDojo only: train (default) | val")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="optional json path for the result")
    args = ap.parse_args()

    stage = load_stage_config(args.config)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    t0 = time.time()
    model = build_model(stage.model)
    report = load_model_weights(model, args.checkpoint)
    model = model.to(device=device, dtype=dtype).eval()
    print(f"[load] {args.checkpoint}: {report if isinstance(report, dict) and len(str(report)) < 400 else 'ok'} "
          f"({time.time() - t0:.0f}s)", flush=True)

    if stage.data.rt2 is not None:
        from metiswam4d.data.rt2 import RT2EpisodeDataset, RT2OnlineEncoder, collate_raw
        dataset = RT2EpisodeDataset(replace(stage.data.rt2, samples_per_episode=1))
        encoder = RT2OnlineEncoder(stage.data.rt2_encoder, device)
        source = "rt2/train"
    else:
        from metiswam4d.data.robodojo import RDJEpisodeDataset
        from metiswam4d.data.rt2 import RT2OnlineEncoder, collate_raw
        cfg = replace(stage.data.robodojo, samples_per_episode=1, fixed_windows=True, split=args.split or "train")
        dataset = RDJEpisodeDataset(cfg)
        encoder = RT2OnlineEncoder(stage.data.robodojo_encoder, device)
        source = f"robodojo/{cfg.split}"
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(dataset), generator=g)[:args.windows].tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=False, collate_fn=collate_raw,
                        num_workers=4)

    schedule = AsyncSchedule(completion={"video": 1.0, "track": 0.5, "action": 0.5},
                             shift={"video": 5.0, "track": 5.0, "action": 5.0})
    mixture = LegacyMixture()
    noise_gen = torch.Generator().manual_seed(args.seed * 31 + 1)
    torch.manual_seed(args.seed)

    sums = {"all": [0.0, 0], "low": [0.0, 0], "mid": [0.0, 0], "high": [0.0, 0]}
    groups = {"xyz": (0, 1, 2, 34, 35, 36), "rot6d": (3, 4, 5, 6, 7, 8, 37, 38, 39, 40, 41, 42), "gripper": (9, 43)}
    gsums = {k: [0.0, 0] for k in groups}
    with torch.no_grad():
        for raw in loader:
            batch = encoder(move_batch(raw, device))
            batch = move_batch(batch, device, dtype)
            step = prepare_training_step(batch, schedule=schedule, mixture=mixture, dropout=None,
                                         generator=noise_gen, dtype=dtype)
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(step.inputs)
            pred, target = out.action_velocity.float(), step.targets["action"].float()
            mask = step.targets.get("action_mask")
            b = pred.shape[0]
            for i in range(b):   # per-sample loss (masked mean over steps x active dims)
                one = torch.ones(1, dtype=torch.bool, device=device)
                li = float(action_loss(pred[i:i + 1], target[i:i + 1], one, mask[i:i + 1] if mask is not None else None))
                s = float(step.assignment.sigma["action"][i])
                sums["all"][0] += li; sums["all"][1] += 1
                key = "low" if s < 1 / 3 else ("mid" if s < 2 / 3 else "high")
                sums[key][0] += li; sums[key][1] += 1
                for name, slots in groups.items():
                    sel = torch.zeros(pred.shape[-1], dtype=torch.bool, device=device); sel[list(slots)] = True
                    m = (mask[i:i + 1] & sel) if mask is not None else sel.expand_as(pred[i:i + 1])
                    gsums[name][0] += float(action_loss(pred[i:i + 1], target[i:i + 1], one, m)); gsums[name][1] += 1
            done = sums["all"][1]
            if done % (args.batch * 8) == 0:
                print(f"  {done}/{args.windows} windows, running mean {sums['all'][0] / done:.5f}", flush=True)

    result = {
        "checkpoint": args.checkpoint, "config": args.config, "source": source, "windows": sums["all"][1],
        "protocol": "legacy: sigma_A = sigma_T = shift5(U), sigma_V independent, no dropout, 20 active dims",
        "action": sums["all"][0] / max(sums["all"][1], 1),
        "bins": {k: (v[0] / v[1] if v[1] else None) for k, v in sums.items() if k != "all"},
        "bin_counts": {k: v[1] for k, v in sums.items() if k != "all"},
        "groups": {k: v[0] / max(v[1], 1) for k, v in gsums.items()},
        "reference": {"janusact4d_rt2_v1_19k-30k": "0.00045-0.00052 (low 0.00106 / mid 0.00069 / high 0.00030)",
                      "janusact4d_rdj_v1_30k-40k": "0.0005-0.0006"},
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
