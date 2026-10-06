"""Training entry for the Memory / Open model: ``metiswam4d.train.train`` with this package's pieces in place of the
RoboDojo defaults (the shared modules are imported, never edited):

    dataset     ``data.robodojo.source = iw0`` -> ``IW0EpisodeDataset`` (canvas, memory frames, joint actions)
    encoder     ``IW0OnlineEncoder`` (memory latents prepended to the window latents)
    model       ``build_iw0_model`` (6-frame clean Video prefix, reader sees the current window)
    init        ``initialize_iw0`` from the ``iw0:`` config section, skipped when the run resumes from a checkpoint
    noise/loss  the whole clean prefix stays clean and is excluded from the Video loss; action diagnostics per joint
                group (arm / gripper) instead of Alpha's EEF groups
    panels      ``build_iw0_visualizer``
    snapshot    ``metiswam4d_inspired_by_internw0/`` joins the run's code archive

    torchrun --nproc_per_node 8 -m metiswam4d_inspired_by_internw0.train --config metiswam4d_inspired_by_internw0/configs/iw0_memopen_v1.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

import metiswam4d.data.robodojo as rdj_data
import metiswam4d.data.rt2 as rt2_data
import metiswam4d.objectives as objectives
import metiswam4d.train.train as base

from metiswam4d_inspired_by_internw0.data import IW0EpisodeDataset, IW0Extras, IW0OnlineEncoder
from metiswam4d_inspired_by_internw0.model import CLEAN_PREFIX, build_iw0_model, initialize_iw0
from metiswam4d_inspired_by_internw0.visualize import build_iw0_visualizer


def install(iw0: dict) -> None:
    original_dataset = rdj_data.window_dataset
    extras = IW0Extras(memory_drop=float(iw0.get("memory_drop", 0.15)))

    def window_dataset(config):
        if config.source == "iw0":
            return IW0EpisodeDataset(config, extras)
        return original_dataset(config)

    rdj_data.window_dataset = window_dataset
    rt2_data.RT2OnlineEncoder = IW0OnlineEncoder
    base.build_model = build_iw0_model

    def initialize_model(model, stage, *, log=print):
        from metiswam4d.train.checkpoint import latest_checkpoint
        if stage.init.resume_from == "auto" and latest_checkpoint(Path(stage.output_dir) / "checkpoints"):
            log("[init] a complete checkpoint exists: weights come from resume")
            return {}
        if not iw0.get("internw0_checkpoint"):
            log("[init] no InternW0 checkpoint configured: fresh weights (smoke)")
            return {}
        return initialize_iw0(model, internw0=iw0["internw0_checkpoint"], alpha=iw0["alpha_checkpoint"],
                              v6=iw0.get("v6_weights"), log=log)

    base.initialize_model = initialize_model
    original_prepare = base.prepare_training_step

    def prepare_training_step(sample, **kwargs):
        step = original_prepare(sample, **kwargs)
        if step.inputs.video is not None:
            clean = sample["video_clean"].to(step.inputs.video.dtype)
            step.inputs.video[:, :, :CLEAN_PREFIX] = clean[:, :, :CLEAN_PREFIX]
        return step

    base.prepare_training_step = prepare_training_step

    def video_loss(pred, target, noisy):
        err = (pred.float() - target.float()).square()[:, :, CLEAN_PREFIX:]
        return objectives._gated_mean(err.mean(dim=(1, 2, 3, 4)), noisy)

    objectives.video_loss = video_loss
    objectives.ACTION_GROUPS = {"arm": (*range(0, 6), *range(40, 46)), "gripper": (16, 56)}   # joint slots
    base.build_visualizer = build_iw0_visualizer
    import metiswam4d.train.snapshot as snapshot
    snapshot.SNAPSHOT_DIRS = (*snapshot.SNAPSHOT_DIRS, "metiswam4d_inspired_by_internw0")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MetisWAM4D Memory / Open trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--stop-after-steps", type=int, default=None)
    args = parser.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text())
    install(raw["iw0"])
    overrides = dict(base._parse_override(item) for item in args.set)
    stage = base.load_stage_config(args.config, overrides)
    base.train(stage, stop_after_steps=args.stop_after_steps)


if __name__ == "__main__":
    main(sys.argv[1:])
