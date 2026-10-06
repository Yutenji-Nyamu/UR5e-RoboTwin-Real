#!/usr/bin/env python3
"""Rewrite gt_action / errors in Efficient-WAM probe outputs produced before the stride-4 GT fix."""
import csv, glob, json, sys
from pathlib import Path
import h5py, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
B = Path("/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin_JanusAct4D_Imperfect/batch_v1")
rows = {(r["task"], r["variant"], r["episode"]): r["source"] for r in csv.DictReader(open(B / "buckets.tsv"), delimiter="\t")}
arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
def errors(pred, gt):
    d = np.abs(pred - gt)
    return dict(pos_cm=float("nan"), rot_deg=float("nan"), joint_deg=float(np.degrees(d[:, arm]).mean()),
                left_joint_deg=float(np.degrees(d[:, :6]).mean()), right_joint_deg=float(np.degrees(d[:, 7:13]).mean()),
                grip=float(0.5 * (d[:, 6].mean() + d[:, 13].mean())), norm_l2=float(np.sqrt(((pred - gt) ** 2).mean())))
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes/p31/effwam")
n = 0
for f in sorted(root.glob("*.npz")):
    z = dict(np.load(f, allow_pickle=True))
    if z.get("gt_stride", None) is not None and int(z["gt_stride"]) == 4:
        continue
    m = json.loads(str(z["meta"])); t, v, name = m["key"].split("/"); s = m["start"]
    with h5py.File(rows[(t, v, name[7:])], "r") as src:
        q = src["joint_action/vector"][:]
    chunk = z["action_full"].shape[0]
    gt = q[np.clip(s + 4 * np.arange(1, chunk + 1), 0, len(q) - 1)].astype(np.float32)
    acts = {k[7:]: z[k] for k in z if k.startswith("action_")}
    errs = {k: dict(vs_gt=errors(a, gt), vs_full=errors(a, acts["full"])) for k, a in acts.items()}
    z["gt_action"] = gt; z["errors"] = json.dumps(errs); z["gt_stride"] = np.int64(4)
    np.savez_compressed(f, **z); n += 1
print("fixed", n)
