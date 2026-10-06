#!/usr/bin/env bash
# Run on each worker machine (t3/t4): high-manipulation subset first, then the rest.
# Safe to re-run: finished shards are skipped, in-flight shards are locked by other drivers.
#   ssh t3 'bash /m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/data_prep/launch_kling_ego_30fps.sh'
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/ytech_milm_intern/danglingwei/datas/KlingHumanEgo20M}"
WORKERS="${WORKERS:-104}"
SCRIPT="$PROJ/scripts/data_prep/kling_ego_30fps.py"
mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/driver_$(hostname -s).out"
cd "$PROJ"
setsid nohup bash -c "
  python $SCRIPT --out-root $OUT_ROOT run --tag hi --score-min 0.5 --workers $WORKERS &&
  python $SCRIPT --out-root $OUT_ROOT run --tag lo --score-max 0.5 --workers $WORKERS
" >> "$LOG" 2>&1 < /dev/null &
sleep 2
echo "launched on $(hostname -s); log: $LOG"
pgrep -f "kling_ego_30fps.py.*run" | wc -l
