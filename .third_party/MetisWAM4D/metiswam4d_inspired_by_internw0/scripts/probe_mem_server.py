"""In-process check of ``MemServer`` on recorded live observations (a teacher recording: 3-view JPEGs upsampled to the
live 640x480, joint states, live head intrinsics / cam2world): begin, two re-plans, close.  With a tiny config and
fresh weights it checks the contract (memory bookkeeping, 32 finite joint targets, grippers in [0, 1]); with a trained
config + exported weights it is the deployment smoke test.

    PYTHONPATH=.:<JanusTrack4d> CUDA_VISIBLE_DEVICES=7 /usr/bin/python3.10 \
        metiswam4d_inspired_by_internw0/scripts/probe_mem_server.py --config <yaml> [--weights <pt>]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from metiswam4d.data.robodojo.episode_dataset import decode_bgr_jpeg

RECORDING = "/ytech_milm_intern/danglingwei/datas/IW0_MemOpen4D/records_smoke/cover_blocks/layout_000.hdf5"


def live_obs(rec, t: int) -> dict:
    vision = {}
    for live, key in (("cam_head", "head_camera"), ("cam_left_wrist", "left_camera"), ("cam_right_wrist", "right_camera")):
        rgb = decode_bgr_jpeg(rec[f"observation/{key}/rgb"][t]).resize((640, 480), Image.BILINEAR)
        vision[live] = {"color": np.asarray(rgb, np.uint8),
                        "intrinsic_matrix": rec[f"observation/{key}/intrinsic_live"][:],
                        "extrinsic_matrix": rec[f"observation/{key}/cam2world_gl"][:]}
    q = rec["joint_state/vector"][t].astype(np.float32)
    return {"vision": vision, "instruction": rec.attrs["instruction"],
            "state": {"left_arm_joint_state": q[0:6], "left_ee_joint_state": q[6:7],
                      "right_arm_joint_state": q[7:13], "right_ee_joint_state": q[13:14]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights")
    args = ap.parse_args()
    weights = args.weights
    if weights is None:
        from metiswam4d.config import load_stage_config
        from metiswam4d_inspired_by_internw0.model import build_iw0_model
        weights = "/tmp/iw0_tiny_weights.pt"
        torch.save(build_iw0_model(load_stage_config(args.config).model).state_dict(), weights)
    from metiswam4d_inspired_by_internw0.eval.mem_server import MemServer
    server = MemServer(args.config, weights, Path("/tmp/iw0_mem_probe"))
    with h5py.File(RECORDING) as rec:
        print("begin", server.begin({"session": "1:cover_blocks:0:0"}))
        out = server.act({"session": "1:cover_blocks:0:0", "acks": [live_obs(rec, 0)]})
        joints = out["joint"]
        print("re-plan 1: step", out["step"], "actions", len(out["actions"]), "seconds", round(out["inference_seconds"], 2))
        assert len(out["actions"]) == 32 and joints.shape == (32, 14) and np.isfinite(joints).all()
        assert joints[:, [6, 13]].min() >= 0 and joints[:, [6, 13]].max() <= 1
        acks = [None] * 31 + [live_obs(rec, 32)]
        out = server.act({"session": "1:cover_blocks:0:0", "acks": acks})
        s = server.sessions["1:cover_blocks:0:0"]
        print("re-plan 2: step", out["step"], "stored canvases", sorted(s["canvases"]), "seconds",
              round(out["inference_seconds"], 2))
        assert out["step"] == 32 and sorted(s["canvases"]) == [0, 32]
        if args.weights:
            nxt = rec["joint_state/vector"][33:65]
            print("mean |pred - recorded next states| per joint:", np.round(np.abs(out["joint"] - nxt).mean(0), 3))
        server.close({"session": "1:cover_blocks:0:0"})
    print("OK")


if __name__ == "__main__":
    main()
