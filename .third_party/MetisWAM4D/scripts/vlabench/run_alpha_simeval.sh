#!/usr/bin/env bash
# VLABench closed-loop campaign on this host (metiswam4d/eval/vlabench_campaign.py).  Run inside tmux:
#   tmux new-session -d -s vlab_alpha_eval "bash scripts/vlabench/run_alpha_simeval.sh <out> [servers per gpu] [gpus] [keep-alive name]"
# - Every CUDA process on the node goes through CUDA MPS (pipe /tmp/nvidia-mps): the policy servers and the GPU
#   keep-alive (wangrunqi_nvml_busy.py) then run their kernels concurrently, so the keep-alive keeps utilisation and
#   memory-bandwidth utilisation above the queue monitor thresholds between policy calls.  MPS clients and non-MPS
#   CUDA processes must not share the node (the non-MPS context starves the MPS ones).  The VLABench clients render
#   with MuJoCo EGL and run torch on CPU, so they hold no CUDA context.
# - A keep-alive that is not an MPS client is stopped (by its pid file) and started again under MPS; one that is
#   missing is started.  It stays on after the campaign.
# - A manager that exits with an error is restarted after 60 s (accepted jobs are kept; up to 10 times).
set -uo pipefail
OUT=$1
SPG=${2:-2}
GPUS=${3:-0,1,2,3,4,5,6,7}
KEEP_NAME=${4:-high16_bdy}
cd "$(dirname "$0")/../.."
LOGS=/m2v_intern_v3/danglingwei/logs
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

if ! pgrep -f "^nvidia-cuda-mps-control" > /dev/null; then
  mkdir -p /tmp/nvidia-mps /tmp/nvidia-mps-log
  nvidia-cuda-mps-control -d && echo "started CUDA MPS control daemon"
fi

keepalive_pid() {
  pgrep -f "^(/usr/local/bin/)?python[0-9.]* /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py" | head -1
}

start_keepalive() {
  (cd /m2v_intern_v3/danglingwei && setsid -f nohup /usr/local/bin/python wangrunqi_nvml_busy.py --gpus all \
     --size 2000 --pid-file "$LOGS/$KEEP_NAME.pid" --log-file "$LOGS/$KEEP_NAME.log" > /dev/null 2>&1 < /dev/null)
  sleep 60
  echo "keep-alive started: pid $(keepalive_pid)"
}

mps_clients() {
  local s
  for s in $(echo get_server_list | nvidia-cuda-mps-control); do echo "get_client_list $s" | nvidia-cuda-mps-control; done
}

pid=$(keepalive_pid)
if [ -n "$pid" ] && ! mps_clients | grep -qx "$pid"; then
  kill "$pid" && echo "stopped non-MPS keep-alive pid $pid"
  sleep 5
  pid=
fi
[ -n "$pid" ] || start_keepalive

# The servers mmap the checkpoint: page faults read Ceph at ~30-250 MB/s, a sequential read ~900 MB/s.  All servers
# on the node then share the page cache.
dd if=/ytech_milm_intern/danglingwei/datas/VLABench/OpenWAM-Alpha-Sim-VLABench/checkpoint_step_6000.safetensors \
   of=/dev/null bs=64M status=none && echo "checkpoint in page cache"

status=1
for attempt in $(seq 1 10); do
  echo "campaign manager attempt $attempt at $(date '+%F %T')"
  /usr/bin/python3.10 -m metiswam4d.eval.vlabench_campaign run --output "$OUT" --gpus "$GPUS" --servers-per-gpu "$SPG"
  status=$?
  echo "campaign manager exited with $status at $(date '+%F %T')"
  [ $status -eq 0 ] && break
  sleep 60
done

[ -n "$(keepalive_pid)" ] || start_keepalive
exit $status
