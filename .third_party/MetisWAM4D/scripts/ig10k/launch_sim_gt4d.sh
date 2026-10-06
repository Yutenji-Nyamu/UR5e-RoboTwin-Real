#!/usr/bin/env bash
# Launch sim_gt4d.py workers on this machine and keep the pod inside its cgroup memory limit.
#   bash scripts/ig10k/launch_sim_gt4d.sh <machine_index> [workers]
# Pods report the host's CPUs / memory in nproc / free; the real budget is the cgroup quota and limit
# (e.g. 24 CPUs / 200 GB on BCC pods).  Default workers = 2/3 of the CPU quota.  A monitor stops the newest worker
# when the working set (usage - inactive_file, the kubelet eviction measure) exceeds MEM_FRAC of the limit; outputs are
# per episode, so a stopped worker loses at most its current episode.  Run inside tmux.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/ig10k/ms_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M=${1:?machine index}
CG=/sys/fs/cgroup/memory
QUOTA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1)
PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
CPUS=$(( QUOTA > 0 ? QUOTA / PERIOD : $(nproc) ))
N=${2:-$(( CPUS * 2 / 3 ))}
MEM_FRAC=${MEM_FRAC:-70}
CG_LIMIT=$(cat $CG/memory.limit_in_bytes)
# MEM_CAP_G caps the budget below the cgroup limit: the node is shared, so pressure can evict the pod earlier.
LIMIT=$CG_LIMIT
[ -n "${MEM_CAP_G:-}" ] && [ $(( MEM_CAP_G << 30 )) -lt "$CG_LIMIT" ] && LIMIT=$(( MEM_CAP_G << 30 ))
OUT=${OUT:-/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/sim}
LOG=$OUT/_logs/workers
mkdir -p "$LOG"
NG=$(nvidia-smi -L | wc -l)
MON=$LOG/$(hostname -s)_monitor.log
echo "$(date '+%F %T') cpus=$CPUS workers=$N cgroup_limit=$((CG_LIMIT >> 30))G budget=$((LIMIT >> 30))G mem_frac=$MEM_FRAC%" | tee -a "$MON"

working_set() {
  local usage inactive
  usage=$(cat $CG/memory.usage_in_bytes)
  inactive=$(awk '/^total_inactive_file /{print $2}' $CG/memory.stat)
  echo $(( usage - inactive ))
}

declare -a PIDS
for i in $(seq 0 $((N - 1))); do
  W=$((M * 1000 + i))
  CUDA_VISIBLE_DEVICES=$((i % NG)) /usr/bin/python3.10 -u scripts/ig10k/sim_gt4d.py --worker "$W" --out "$OUT" \
    >> "$LOG/$(hostname -s)_w$(printf %03d $i).log" 2>&1 &
  PIDS+=($!)
  sleep 5
done

while :; do
  alive=()
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
  [ ${#alive[@]} -eq 0 ] && break
  ws=$(working_set)
  echo "$(date '+%F %T') alive=${#alive[@]} working_set=$((ws >> 30))G limit=$((LIMIT >> 30))G" >> "$MON"
  if [ $(( ws * 100 / LIMIT )) -ge "$MEM_FRAC" ]; then
    victim=${alive[-1]}
    echo "$(date '+%F %T') working set above ${MEM_FRAC}%: stopping worker pid $victim" | tee -a "$MON"
    kill "$victim"
    PIDS=("${alive[@]:0:${#alive[@]}-1}")
  fi
  sleep 20
done
echo "$(date '+%F %T') all workers on $(hostname -s) exited" | tee -a "$MON"
