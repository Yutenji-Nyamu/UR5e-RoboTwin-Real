#!/usr/bin/env bash
# Extra generate_4d.py workers for slow tasks: helper w starts in the second half of launch_generate.sh worker w's
# range (w * PER + PER / 2); the original worker skips the files the helper has already written.
#   bash scripts/vlabench/launch_helpers.sh add_condiment insert_flower
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-/ytech_milm_intern/danglingwei/datas/VLABench/gen4d}
EPISODES=${EPISODES:-500}
WORKERS_PER_TASK=${WORKERS_PER_TASK:-10}
NGPU=$(nvidia-smi -L | wc -l)
PER=$(( EPISODES / WORKERS_PER_TASK ))
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
i=0
for task in "$@"; do
  for w in $(seq 0 $(( WORKERS_PER_TASK - 1 ))); do
    start=$(( w * PER + PER / 2 ))
    MUJOCO_EGL_DEVICE_ID=$(( i % NGPU )) /usr/local/vlabench/venv/bin/python scripts/vlabench/generate_4d.py \
      --task "$task" --out "$OUT" --start "$start" --count $(( PER - PER / 2 )) \
      > "$OUT/_logs/stdout_${task}_helper${w}.log" 2>&1 &
    i=$(( i + 1 ))
  done
done
echo "[$(date '+%F %T')] launched $i helpers"
wait
echo "[$(date '+%F %T')] helpers finished"
