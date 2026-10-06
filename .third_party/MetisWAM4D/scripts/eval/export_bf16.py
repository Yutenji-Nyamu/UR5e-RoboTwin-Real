"""DCP checkpoint(s) -> one bf16 ``state_dict`` file (fast repeated loading by the evaluation servers).

    PYTHONPATH=. /usr/bin/python3.10 scripts/eval/export_bf16.py --config configs/rt2_direct_v3_coupled.yaml \
        --checkpoint <run>/checkpoints/save_step_0033760 --out <eval dir>/model_bf16.pt

Several comma-separated checkpoints are averaged uniformly in fp32 (tail-of-training weight averaging); buffers and
integer tensors are taken from the last one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from metiswam4d.build import build_model
from metiswam4d.config import load_stage_config
from metiswam4d.train.checkpoint import load_model_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="one DCP directory, or several comma-separated to average")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--builder", choices=["metiswam4d", "iw0"], default="metiswam4d",
                    help="iw0: the Memory / Open model (metiswam4d_inspired_by_internw0.model.build_iw0_model)")
    args = ap.parse_args()
    started = time.time()
    checkpoints = args.checkpoint.split(",")
    if args.builder == "iw0":
        from metiswam4d_inspired_by_internw0.model import build_iw0_model as build
    else:
        build = build_model
    model = build(load_stage_config(args.config).model)
    reports, mean = [], None
    for i, checkpoint in enumerate(checkpoints):
        report = load_model_weights(model, checkpoint)
        if report["missing"] or report["unexpected"]:
            raise RuntimeError(f"checkpoint {checkpoint} does not cover the model: {report}")
        reports.append({"checkpoint": checkpoint, "loaded": report["loaded"]})
        state = model.state_dict()
        if len(checkpoints) == 1:
            mean = state
            break
        if mean is None:
            mean = {k: v.float().clone() if v.is_floating_point() else v.clone() for k, v in state.items()}
        else:
            for k, v in state.items():
                if v.is_floating_point():
                    mean[k].add_(v.float())
                else:
                    mean[k] = v.clone()
        print(json.dumps({"averaged": i + 1, "checkpoint": checkpoint, "seconds": round(time.time() - started)}), flush=True)
    if len(checkpoints) > 1:
        for k, v in mean.items():
            if v.is_floating_point():
                v.div_(len(checkpoints))
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in mean.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(args.out)
    info = {"source": checkpoints if len(checkpoints) > 1 else checkpoints[0], "average": len(checkpoints) > 1,
            "tensors": len(state), "load_report": reports, "seconds": round(time.time() - started)}
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
