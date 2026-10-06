#!/usr/bin/env bash
# Log the pod's cgroup-v1 memory breakdown and per-worker RSS every 20 s (diagnoses where sim_gt4d memory goes).
#   bash scripts/ig10k/mem_probe.sh <out.tsv>
CG=/sys/fs/cgroup/memory
OUT=${1:?out tsv}
echo -e "time\tusage_G\tkmem_G\ttotal_rss_G\ttotal_cache_G\ttotal_inactive_file_G\ttotal_shmem_G\ttotal_mapped_file_G\tworkers\tworker_rss_sum_G\tworker_rss_max_G\tgpu_mem_sum_G" > "$OUT"
while :; do
  rss=""
  for p in $(pgrep -f "^/usr/bin/python3.10 -u scripts/ig10k/sim_gt4d.py"); do
    rss="$rss $(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null)"
  done
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')
  awk -v t="$(date +%T)" -v usage="$(cat $CG/memory.usage_in_bytes)" -v gpu="$gpu" \
      -v kmem="$(cat $CG/memory.kmem.usage_in_bytes 2>/dev/null || echo 0)" -v rss="$rss" '
    { s[$1] = $2 }
    END {
      G = 2^30; n = split(rss, r, " "); sum = 0; max = 0
      for (i = 1; i <= n; i++) { sum += r[i]; if (r[i] > max) max = r[i] }
      printf "%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%.2f\t%.2f\t%.2f\n", t, usage / G, kmem / G,
        s["total_rss"] / G, s["total_cache"] / G, s["total_inactive_file"] / G, s["total_shmem"] / G,
        s["total_mapped_file"] / G, n, sum / 2^20, max / 2^20, gpu / 1024
    }' $CG/memory.stat >> "$OUT"
  sleep 20
done
