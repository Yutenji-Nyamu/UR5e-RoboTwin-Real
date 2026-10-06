"""Build the production-size model on CPU and apply ``initialize_iw0``; print the load report and spot-check tensors.

    PYTHONPATH=. /usr/bin/python3.10 metiswam4d_inspired_by_internw0/scripts/check_init.py --config <yaml>
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from metiswam4d.config import load_stage_config
from metiswam4d_inspired_by_internw0.model import build_iw0_model, initialize_iw0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    iw0 = yaml.safe_load(Path(args.config).read_text())["iw0"]
    stage = load_stage_config(args.config)
    model = build_iw0_model(stage.model).to(torch.bfloat16)
    report = initialize_iw0(model, internw0=iw0["internw0_checkpoint"], alpha=iw0["alpha_checkpoint"],
                            v6=iw0.get("v6_weights"))
    counts = {k: f"{sum(p.numel() for p in v) / 1e6:.1f}M" for k, v in model.trainable_groups().items()}
    print("[params]", counts)
    mot = torch.load(iw0["internw0_checkpoint"], map_location="cpu", mmap=True, weights_only=False)["mot"]
    checks = {
        "video.blocks.7.self_attn.q.weight": "mixtures.video.blocks.7.self_attn.q.weight",
        "action.blocks.7.cross_attn.q.weight": "mixtures.action.blocks.7.cross_attn.q.weight",
        "action.action_decoder.weight": "mixtures.action.head.weight",
    }
    own = model.state_dict()
    for ours, theirs in checks.items():
        print(f"[check] {ours} == {theirs}: {torch.equal(own[ours].float(), mot[theirs].float())}")
    print("[check] video clean prefix:", model.video.config.clean_frames)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else vv[:8]) for kk, vv in v.items()}
                      if isinstance(v, dict) else v for k, v in report.items()}, default=str, indent=1)[:4000])


if __name__ == "__main__":
    main()
