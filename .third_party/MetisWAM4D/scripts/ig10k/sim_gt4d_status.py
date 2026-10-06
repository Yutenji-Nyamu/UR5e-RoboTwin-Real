"""Progress of scripts/ig10k/sim_gt4d.py from its logs (no directory scans of the dataset)."""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/sim")
recs = []
for f in glob.glob(str(out / "_logs" / "*.jsonl")):
    if Path(f).name.startswith("bad_"):
        continue
    for line in open(f):
        recs.append(json.loads(line))
ok = [r for r in recs if r["ok"]]
bad = [r for r in recs if not r["ok"]]
done = len(list((out / "_done").glob("*"))) if (out / "_done").exists() else 0
locks = len(list((out / "_locks").glob("*"))) if (out / "_locks").exists() else 0
print(f"episodes ok {len(ok)}  failed {len(bad)}  dirs done {done}/200  dirs in progress {locks}")
if ok:
    t = sorted(r["time"] for r in ok)
    print(f"first {t[0]}  last {t[-1]}  mean t_total {sum(r['t_total'] for r in ok) / len(ok):.1f}s  "
          f"land<5mm median {sorted(r['land_within_5mm'] for r in ok)[len(ok) // 2]:.3f}  "
          f"qpos maxerr p99 {sorted(r['qpos_maxerr'] for r in ok)[int(0.99 * (len(ok) - 1))]:.4f}")
    print("hosts", dict(Counter(r["host"].split(".")[0] for r in ok)))
if bad:
    print("failures by reason", dict(Counter(r.get("reason") for r in bad)))
    print("failures by dir", dict(Counter(r["env_dir"] for r in bad).most_common(10)))
