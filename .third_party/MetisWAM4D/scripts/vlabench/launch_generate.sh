#!/usr/bin/env bash
# Generate VLABench primitive tasks (EPISODES each; TASK_LIST overrides the 10) with generate_4d.py on one node.
#   tmux new -d -s vlab_gen "bash scripts/vlabench/launch_generate.sh"
# WORKERS_PER_TASK processes per task, each producing EPISODES / WORKERS_PER_TASK episodes; MuJoCo EGL rendering is
# spread over the node's GPUs.  Re-running resumes (finished episode files are skipped).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-/ytech_milm_intern/danglingwei/datas/VLABench/gen4d}
EPISODES=${EPISODES:-500}
WORKERS_PER_TASK=${WORKERS_PER_TASK:-10}
START_BASE=${START_BASE:-0}   # first episode index (seeds follow the index, so a new range gives new scenes)
NGPU=$(nvidia-smi -L | wc -l)
read -r -a TASKS <<< "${TASK_LIST:-add_condiment insert_flower select_book select_chemistry_tube select_drink select_fruit select_mahjong select_painting select_poker select_toy}"
PER=$(( EPISODES / WORKERS_PER_TASK ))
mkdir -p "$OUT/_logs"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
i=0
for task in "${TASKS[@]}"; do
  for w in $(seq 0 $(( WORKERS_PER_TASK - 1 ))); do
    # IDC nodes enumerate only the CUDA-visible GPU as an EGL device
    CUDA_VISIBLE_DEVICES=$(( i % NGPU )) MUJOCO_EGL_DEVICE_ID=0 /usr/local/vlabench/venv/bin/python scripts/vlabench/generate_4d.py \
      --task "$task" --out "$OUT" --start $(( START_BASE + w * PER )) --count "$PER" \
      > "$OUT/_logs/stdout_${task}_${w}.log" 2>&1 &
    i=$(( i + 1 ))
  done
done
echo "[$(date '+%F %T')] launched $i workers"
wait
echo "[$(date '+%F %T')] all workers finished"
