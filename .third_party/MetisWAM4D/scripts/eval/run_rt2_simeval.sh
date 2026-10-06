#!/usr/bin/env bash
# One host of the RT2 closed-loop campaign (metiswam4d/eval/rt2_campaign.py).  Run inside tmux:
#   tmux new-session -d -s rt2eval "bash scripts/eval/run_rt2_simeval.sh <out[,out2]> <gpus> <sims> <keepalive pid file> <keepalive name>"
# - CUDA MPS is started first: the model server and the simulation processes (cuRobo planning every step) then run
#   their kernels concurrently instead of time-slicing the GPU.
# - The GPU keep-alive (wangrunqi_nvml_busy.py) is the load watchdog of the queue monitor: it only fills the
#   utilisation / memory gap the real job leaves.  The instance started outside MPS is stopped (it would time-slice
#   against the campaign) and started again as an MPS client (new pid / log file) before the campaign; it stays on
#   through the campaign and afterwards.  If it died meanwhile it is started again when the campaign is over.
# - A manager that exits with an error is restarted after 60 s (episode-level resume; up to 20 times).
set -uo pipefail
OUT=$1
GPUS=$2
SIMS=${3:-6}
KEEP_PID=${4:-}
KEEP_NAME=${5:-}
cd "$(dirname "$0")/../.."

stop_keepalive() {
  local file=$1
  [ -n "$file" ] && [ -f "$file" ] || return 0
  local pid
  pid=$(sed -n 's/^pid=//p' "$file")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "stopped keep-alive pid $pid ($file)"
    sleep 5
  fi
}

start_keepalive() {
  [ -n "$KEEP_NAME" ] || return 0
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  (cd /m2v_intern_v3/danglingwei && setsid -f nohup python wangrunqi_nvml_busy.py --gpus all --size 2000 \
     --pid-file "logs/${KEEP_NAME}_${stamp}.pid" --log-file "logs/${KEEP_NAME}_${stamp}.log" \
     > /dev/null 2>&1 < /dev/null)
  for _ in $(seq 1 30); do  # the pid file is written once all GPUs are initialised (up to ~1 min on 8 GPUs)
    KEEP_PID=$(ls -t /m2v_intern_v3/danglingwei/logs/${KEEP_NAME}_${stamp}*.pid 2>/dev/null | head -1)
    [ -n "$KEEP_PID" ] && break
    sleep 3
  done
  echo "keep-alive started: ${KEEP_PID:-<pid file not found>}"
}

keepalive_alive() {
  [ -n "$KEEP_PID" ] && [ -f "$KEEP_PID" ] && kill -0 "$(sed -n 's/^pid=//p' "$KEEP_PID")" 2>/dev/null
}

stop_keepalive "$KEEP_PID"

if ! pgrep -x nvidia-cuda-mps > /dev/null && ! pgrep -f "^nvidia-cuda-mps-control" > /dev/null; then
  mkdir -p /tmp/nvidia-mps /tmp/nvidia-mps-log
  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log nvidia-cuda-mps-control -d \
    && echo "started CUDA MPS control daemon"
fi
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

start_keepalive

status=1
for attempt in $(seq 1 20); do
  echo "campaign manager attempt $attempt at $(date '+%F %T')"
  /usr/bin/python3.10 -m metiswam4d.eval.rt2_campaign run --output "$OUT" --gpus "$GPUS" --sims "$SIMS"
  status=$?
  echo "campaign manager exited with $status at $(date '+%F %T')"
  [ $status -eq 0 ] && break
  sleep 60
done

if [ -n "$KEEP_NAME" ] && ! keepalive_alive; then
  echo "keep-alive not running after the campaign; starting it again"
  start_keepalive
fi
exit $status
