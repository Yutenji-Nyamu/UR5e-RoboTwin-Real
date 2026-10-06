"""Offline probe of a running InternW0 server: a real RoboDojo frame (RDJ episode, decoded to RGB and upsampled to the
640x480 Isaac resolution) plus its joint state -> begin / act x2 / close; prints the returned joint chunk next to the
recorded next states.

    PYTHONPATH=. /usr/bin/python3.10 metiswam4d_inspired_by_internw0/scripts/probe_iw0_server.py --port 35999
"""
import argparse
import sys

import h5py
import numpy as np
from PIL import Image

ROBODOJO = "/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo"
EPISODE = "/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D/cover_blocks/train_4d/episode0"


def obs_at(src, t: int) -> dict:
    from metiswam4d.data.robodojo.episode_dataset import decode_bgr_jpeg
    vision = {}
    for cam, key in (("cam_head", "head_camera"), ("cam_left_wrist", "left_camera"), ("cam_right_wrist", "right_camera")):
        rgb = decode_bgr_jpeg(src[f"observation/{key}/rgb"][t]).resize((640, 480), Image.BILINEAR)
        vision[cam] = {"color": np.asarray(rgb, np.uint8)}
    q = src["joint_state/vector"][t].astype(np.float32)
    state = {"left_arm_joint_state": q[0:6], "left_ee_joint_state": q[6:7], "right_arm_joint_state": q[7:13],
             "right_ee_joint_state": q[13:14]}
    return {"vision": vision, "state": state, "instruction": src["metadata/instruction"][()].decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    sys.path[:0] = [ROBODOJO, ROBODOJO + "/XPolicyLab"]
    from client_server.ws.model_client import WsModelClient
    client = WsModelClient(url=f"ws://127.0.0.1:{args.port}", evaluation_id="iw0_probe", trial_id="probe")
    with h5py.File(EPISODE + "/source.hdf5") as src:
        info = client.call(func_name="begin", obs={"session": "probe"})
        print("begin:", info)
        out = client.call(func_name="act", obs={"session": "probe", "acks": [obs_at(src, 0)]})
        joint = np.asarray(out["joint"])
        nxt = src["joint_state/vector"][1:1 + len(joint)]
        print("chunk", joint.shape, "seconds", round(out["inference_seconds"], 2), "step", out["step"])
        print("pred[0] ", np.round(joint[0], 3))
        print("gt  [1] ", np.round(nxt[0], 3))
        print("mean |pred - recorded next states| per joint:", np.round(np.abs(joint - nxt).mean(0), 3))
        acks = [None] * (len(joint) - 1) + [obs_at(src, len(joint))]
        out2 = client.call(func_name="act", obs={"session": "probe", "acks": acks})
        print("second act: step", out2["step"], "seconds", round(out2["inference_seconds"], 2))
        client.call(func_name="close", obs={"session": "probe"})


if __name__ == "__main__":
    main()
