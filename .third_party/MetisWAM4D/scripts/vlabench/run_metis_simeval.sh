#!/usr/bin/env bash
# MetisWAM4D VLABench campaign on this host (metiswam4d/eval/vlabench_campaign.py --backend metis), inside tmux:
#   tmux new-session -d -s <name> "bash scripts/vlabench/run_metis_simeval.sh <out> <model_bf16.pt> <gpus> <port base> [extra campaign args]"
# Every CUDA process on the node goes through CUDA MPS (as in run_alpha_simeval.sh): the CUDA MPS daemon is started if
# absent, and a keep-alive that is not an MPS client is stopped through its own pid file and started again with the same
# arguments under MPS, so it runs concurrently with the policy servers instead of time-slicing them.
# A manager that exits with an error is restarted after 60 s (accepted jobs are kept; up to 10 times).
set -uo pipefail
OUT=$1
MODEL=$2
GPUS=$3
PORT_BASE=$4
shift 4
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
keep_re='^(/usr/local/bin/)?python[0-9.]* /(m2v_intern_v3|ytech_milm_intern)/danglingwei/wangrunqi_nvml_busy.py'
if ! pgrep -f '^nvidia-cuda-mps-control' >/dev/null; then
  mkdir -p /tmp/nvidia-mps /tmp/nvidia-mps-log
  nvidia-cuda-mps-control -d && echo "started CUDA MPS control daemon" >> "$OUT/manager.log"
  /usr/bin/python3.10 -c "import torch; torch.ones(1, device='cuda')"   # starts the MPS server
fi
mps_clients() { local s; for s in $(echo get_server_list | nvidia-cuda-mps-control); do echo "get_client_list $s" | nvidia-cuda-mps-control; done; }
keep_pid=$(pgrep -f "$keep_re" | head -1)
if [[ -n "$keep_pid" ]] && ! mps_clients | grep -qx "$keep_pid"; then
  mapfile -d '' args < "/proc/$keep_pid/cmdline"
  base=''
  for i in "${!args[@]}"; do [[ ${args[$i]} == --pid-file ]] && base=${args[$((i + 1))]}; done
  for f in "$base" "${base%.pid}"_*.pid; do
    [[ -f "$f" && $(sed -n 's/^pid=//p' "$f") == "$keep_pid" ]] || continue
    kill -TERM "$keep_pid"
    for _ in $(seq 60); do kill -0 "$keep_pid" 2>/dev/null || break; sleep 1; done
    setsid -f nohup /usr/local/bin/python "${args[@]:1}" > /dev/null 2>&1 < /dev/null
    echo "keep-alive pid $keep_pid restarted as an MPS client" >> "$OUT/manager.log"
    break
  done
elif [[ -z "$keep_pid" ]]; then
  setsid -f nohup /usr/local/bin/python /ytech_milm_intern/danglingwei/wangrunqi_nvml_busy.py --gpus all --size 2000 \
    --pid-file /ytech_milm_intern/danglingwei/logs/high16_bdy.pid --log-file /ytech_milm_intern/danglingwei/logs/high16_bdy.log \
    > /dev/null 2>&1 < /dev/null
  echo "keep-alive started as an MPS client" >> "$OUT/manager.log"
fi
git -C . status --short > "$OUT/code_state_$(date +%Y%m%d_%H%M%S).txt" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for attempt in $(seq 1 10); do
  /usr/bin/python3.10 -m metiswam4d.eval.vlabench_campaign run --output "$OUT" --backend metis --model-file "$MODEL" \
    --gpus "$GPUS" --servers-per-gpu 2 --port-base "$PORT_BASE" "$@" >> "$OUT/manager.log" 2>&1
  status=$?
  [[ $status -eq 0 ]] && break
  echo "[$(date '+%F %T')] manager exited with $status (attempt $attempt); restarting in 60 s" >> "$OUT/manager.log"
  sleep 60
done
/usr/bin/python3.10 -m metiswam4d.eval.vlabench_campaign summary --output "$OUT" > "$OUT/summary_final.md" 2>&1
