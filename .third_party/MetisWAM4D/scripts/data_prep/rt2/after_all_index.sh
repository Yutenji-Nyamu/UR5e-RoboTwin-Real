#!/usr/bin/env bash
# Run on any node: once every shard log of the current launch reports "shard finished", rebuild
# index.jsonl over all variants and print the final status.  The norm / depth statistics stay frozen
# (computed from the clean set by after_clean_then_randomized.sh) so the mu-law scale is stable.
#   bash scripts/data_prep/rt2/after_all_index.sh "2026-09-22 14:09"    # launch time of the shard logs
set -uo pipefail
PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT"
PY=/usr/bin/python3.10
LOGS=/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D/_logs
since="${1:?launch time, e.g. '2026-09-22 14:09'}"
while true; do
  total=0; finished=0
  while IFS= read -r f; do
    total=$((total + 1))
    grep -q "shard finished" "$f" && finished=$((finished + 1))
  done < <(find "$LOGS" -maxdepth 1 -name "*rt2track_s*.out" -newermt "$since")
  echo "$(date '+%F %T') shards finished: $finished / $total"
  [[ "$total" -gt 0 && "$finished" -ge "$total" ]] && break
  sleep 300
done
# Sweep: episodes that FAILED in this launch (e.g. poses.npz truncated by the t4 reboot) are retried once
# by a single shard on SWEEP_NODE with the patched script; the list comes from the logs (no directory scan).
OUT=/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D
python3 - "$since" "$LOGS" "$OUT" <<'PYEOF'
import glob, json, os, re, sys
since, logs, out = sys.argv[1], sys.argv[2], sys.argv[3]
failed, done = set(), set()
for f in glob.glob(f"{logs}/*rt2track_s*.out") + glob.glob(f"{logs}/dead_*/*.out"):
    for line in open(f, errors="ignore"):
        m = re.match(r"\[(\S+ \S+)\] (FAILED|done) (\S+)/(\S+)/episode(\d+)", line)
        if m and m.group(1) >= since:
            (failed if m.group(2) == "FAILED" else done).add((m.group(3), m.group(4), int(m.group(5))))
rows = sorted(failed - done)
with open(f"{out}/pending.jsonl", "w") as f:
    for task, variant, ep in rows:
        f.write(json.dumps({"task": task, "variant": variant, "episode": ep}) + "\n")
print(f"sweep: {len(rows)} failed episodes -> pending.jsonl")
PYEOF
if [[ -s "$OUT/pending.jsonl" ]]; then
  ssh -o BatchMode=yes "${SWEEP_NODE:-t3}" "cd $PROJECT && CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 $PY scripts/data_prep/rt2/rt2_track4d.py run --shard 0 --num-shards 1 --device cuda:0 --variants demo_clean_4d demo_randomized_4d" \
    > "$LOGS/sweep_failed.out" 2>&1
  echo "$(date '+%F %T') sweep done: $(grep -c '] done' "$LOGS/sweep_failed.out") recovered, $(grep -c FAILED "$LOGS/sweep_failed.out") still failing"
fi
$PY scripts/data_prep/rt2/rt2_track4d.py index --variants demo_clean_4d demo_randomized_4d
$PY scripts/data_prep/rt2/rt2_track4d.py status --variants demo_clean_4d demo_randomized_4d
echo "$(date '+%F %T') final index done"
