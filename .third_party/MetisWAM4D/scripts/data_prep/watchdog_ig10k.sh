#!/usr/bin/env bash
# Keep the IG-10K annotation workers alive so GPU utilisation never silently drops to zero
# (an idle node risks being reclaimed by the cluster monitor).
#
# Every INTERVAL seconds: count live workers, relaunch the shortfall, log util + worker count.
#   PROFILE=robot GPUS="0 1" bash .../watchdog_ig10k.sh start          # depth+masks stage
#   PROFILE=robot STAGE=track4d GPUS="0 1" bash .../watchdog_ig10k.sh start
#   bash .../watchdog_ig10k.sh stop | status
# One watchdog per (host, stage): the pid file is keyed by STAGE.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${PROFILE:-human}"
STAGE="${STAGE:-depth_masks}"            # depth_masks | track4d
# depth_masks only: restrict workers to dirs whose object masks are shipped (19 GB, 3/GPU) or need
# SAM (30 GB, 2/GPU).  Keep one kind per host: the worker count below is per host.
MASK_SOURCE="${MASK_SOURCE:-}"           # "" | shipped | sam
OUT_ROOT="${OUT_ROOT:-/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/${PROFILE}}"
GPUS="${GPUS:-0 1 2 3 4 5}"
PER_GPU="${PER_GPU:-2}"
INTERVAL="${INTERVAL:-180}"
PY=/usr/bin/python3.10
HOST="$(hostname -s)"
LOG="$OUT_ROOT/_logs/watchdog_${STAGE}_${HOST}.log"
PIDFILE="/tmp/ig10k_watchdog_${STAGE}.pid"

case "$STAGE" in
depth_masks) SCRIPT=scripts/data_prep/ig10k_human_depth_masks.py; ARGS="--profile $PROFILE run --shuffle" ;;
track4d)     SCRIPT=scripts/data_prep/ig10k_human_track4d.py;     ARGS="--profile $PROFILE run --shuffle" ;;
*) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
[ -n "$MASK_SOURCE" ] && ARGS="$ARGS --mask-source $MASK_SOURCE"
# anchored on the interpreter so the pattern can never match a shell that merely mentions the script
PATTERN="^$PY $SCRIPT --profile $PROFILE run"

n_workers() { pgrep -fc -- "$PATTERN" || true; }
# workers pinned to one GPU, read from each worker's CUDA_VISIBLE_DEVICES (replacements must land on
# the GPU that lost a worker, not pile up on the first one in GPUS)
n_on_gpu() {
  local g="$1" n=0 p
  for p in $(pgrep -f -- "$PATTERN"); do
    [ "$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')" = "$g" ] && n=$((n + 1))
  done
  echo $n
}

case "${1:-start}" in
stop)
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE"
  echo "watchdog[$STAGE] stopped on $HOST"
  exit 0
  ;;
status)
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "watchdog[$STAGE] running on $HOST (pid $(cat "$PIDFILE")), workers=$(n_workers)"
  else
    echo "watchdog[$STAGE] NOT running on $HOST, workers=$(n_workers)"
  fi
  tail -3 "$LOG" 2>/dev/null
  exit 0
  ;;
esac

want=0
for _ in $GPUS; do want=$((want + PER_GPU)); done

mkdir -p "$OUT_ROOT/_logs"
loop() {
  cd "$PROJ" || exit 1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  while true; do
    have=$(n_workers)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd,)
    if [ "$have" -lt "$want" ]; then
      # top each GPU up to PER_GPU
      for g in $GPUS; do
        cur=$(n_on_gpu "$g")
        while [ "$cur" -lt "$PER_GPU" ]; do
          # shellcheck disable=SC2086
          CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" "$SCRIPT" $ARGS \
            >> "$OUT_ROOT/_logs/${STAGE}_driver_${HOST}_gpu${g}.out" 2>&1 < /dev/null &
          cur=$((cur + 1))
          sleep 2
        done
      done
      echo "[$(date '+%m-%d %H:%M:%S')] $HOST had $have/$want workers, relaunched -> $(n_workers) (util $util)" >> "$LOG"
    else
      echo "[$(date '+%m-%d %H:%M:%S')] $HOST workers=$have/$want util=$util" >> "$LOG"
    fi
    sleep "$INTERVAL"
  done
}

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "watchdog[$STAGE] already running on $HOST (pid $(cat "$PIDFILE"))"
  exit 0
fi
loop &
echo $! > "$PIDFILE"
disown
echo "watchdog[$STAGE] started on $HOST (pid $(cat "$PIDFILE")), profile $PROFILE${MASK_SOURCE:+ mask-source $MASK_SOURCE}, GPUs [$GPUS] x $PER_GPU = $want workers, every ${INTERVAL}s"
