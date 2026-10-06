#!/usr/bin/env bash
# Full local EBench evaluation of an OpenWAM-style checkpoint on one node:
#   policy servers (one per GPU, port 8848 + gpu, tmux ebench_alpha) -> GenManip server (tmux ebench_eval:server,
#   WORKERS_PER_GPU Isaac workers per GPU) -> submit SPLIT -> bridge workers sharing the policy servers.
# Every CUDA process on the node must be on the same side of the platform's MPS daemon (MPS clients next to plain
# CUDA processes barely get to run, which starved the keepalive). EBENCH_USE_MPS=1 runs policy servers and Isaac
# workers in MPS (keepalive started normally); otherwise all stay out of it and the keepalive must have been started
# with CUDA_MPS_PIPE_DIRECTORY=/tmp/ebench-no-mps.
#   RUN_ID=alpha_testmini SPLIT=ebench/generalist/test_mini bash scripts/ebench/launch_alpha_eval.sh
set -euo pipefail
: "${RUN_ID:?}"
SPLIT=${SPLIT:-ebench/generalist/test_mini}
CKPT_DIR=${CKPT_DIR:-/ytech_milm_intern/danglingwei/datas/EBench/OpenWAM-Alpha-Sim-EBench}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=/m2v_intern_v3/danglingwei/logs/ebench
OPENWAM=/ytech_milm_intern/danglingwei/files/EBench_suite/OpenWAM
GENMANIP=/ytech_milm_intern/danglingwei/files/EBench_suite/GenManip
read -r -a gpu_list <<< "$GPUS"
num_gpu=${#gpu_list[@]}

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if [[ ${EBENCH_USE_MPS:-0} == 1 ]]; then mps_env=""; else mps_env="CUDA_MPS_PIPE_DIRECTORY=/tmp/ebench-no-mps"; fi

# 1. policy servers
tmux has-session -t ebench_alpha 2>/dev/null || tmux new-session -d -s ebench_alpha -n idle
for g in "${gpu_list[@]}"; do
  port=$((8848 + g))
  port_open "$port" && continue
  tmux new-window -t ebench_alpha -n "gpu$g" "cd $OPENWAM && $mps_env \
    OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=. /usr/bin/python3.10 scripts/deploy.py --ckpt-dir $CKPT_DIR --device cuda:$g --port $port \
    --compile-enabled false optimization.dit_cache.enabled=false 2>&1 | tee $LOGS/policy_gpu$g.log"
done
for g in "${gpu_list[@]}"; do
  until port_open $((8848 + g)); do sleep 10; done
done
echo "policy servers ready on ports $(for g in "${gpu_list[@]}"; do printf '%s ' $((8848 + g)); done)"

# 2. GenManip server (restarted so the worker packing and render settings apply)
tmux kill-session -t ebench_eval 2>/dev/null || true
sleep 5
gpu_mem_gib=$(nvidia-smi -i "${gpu_list[0]}" --query-gpu=memory.total --format=csv,noheader,nounits)
job_mem=$(awk -v m="$gpu_mem_gib" -v k="$WORKERS_PER_GPU" 'BEGIN { printf "%.1f", m / 1024 / k - 0.5 }')
: > $LOGS/genmanip_server.log
tmux new-session -d -s ebench_eval -n server "EBENCH_USE_MPS=${EBENCH_USE_MPS:-0} GENMANIP_JOB_MEMORY_GB=$job_mem bash $HERE/run_genmanip_server.sh \
  2>&1 | tee $LOGS/genmanip_server.log"
until grep -q "Uvicorn running" $LOGS/genmanip_server.log; do sleep 3; done

# 3. submit
(cd "$GENMANIP" && /usr/local/ebench/genmanip-venv/bin/gmp submit "$SPLIT" --run_id "$RUN_ID")

# 4. bridge workers
num_workers=$((num_gpu * WORKERS_PER_GPU))
tmux new-window -t ebench_eval -n bridge "RUN_ID=$RUN_ID WORKERS='$(seq -s ' ' 0 $((num_workers - 1)))' \
  NUM_POLICY=$num_gpu SOUTH_PORT_BASE=$((8848 + gpu_list[0])) bash $HERE/run_alpha_bridge.sh"
echo "launched $num_workers workers for $RUN_ID ($SPLIT); logs $LOGS/$RUN_ID"
