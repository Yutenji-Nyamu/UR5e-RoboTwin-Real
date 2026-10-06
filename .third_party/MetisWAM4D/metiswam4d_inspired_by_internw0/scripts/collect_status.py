"""Teacher collection status: per task successes / episodes over the campaigns, converted 4D episodes, GPU load.

    /usr/bin/python3.10 metiswam4d_inspired_by_internw0/scripts/collect_status.py
"""
import glob
import json
from pathlib import Path
import subprocess

OUT = Path("/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/iw0_teacher")
DATA = Path("/ytech_milm_intern/danglingwei/datas/IW0_MemOpen4D/dataset_v1")
CAMPAIGNS = ("collect_t1_seed1", "collect_t1_seed2", "collect_t1_seed0",
             "collect_t1_seed1b", "collect_t1_seed2b", "collect_t1_seed0b")


def main():
    total, target = {}, {}
    for c in CAMPAIGNS:
        camp = json.loads((OUT / c / "campaign.json").read_text())
        for v in camp["variants"]:
            target[v["task"]] = target.get(v["task"], 0) + v["episodes"]
        for p in glob.glob(str(OUT / c / "results/RoboDojo/*/**/_result.json"), recursive=True):
            d = json.load(open(p))
            task = p.split("/RoboDojo/")[1].split("/")[0]
            det = (d.get("details") or {}).values()
            t = total.setdefault(task, [0, 0])
            t[0] += sum(1 for v in det if v.get("success"))
            t[1] += len(det)
    converted = {}
    for meta in DATA.glob("*/rollout_t1/episode*/meta.json"):
        task = meta.parts[-4]
        converted[task] = converted.get(task, 0) + 1
    for task in sorted(target):
        s, n = total.get(task, [0, 0])
        print(f"{task:32s} success {s:4d} / done {n:4d} / target {target[task]:4d}   converted {converted.get(task, 0)}")
    done = sum(n for _, n in total.values())
    print(f"episodes {done} / {sum(target.values())}")
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.replace("\n", " | "))


if __name__ == "__main__":
    main()
