#!/usr/bin/env bash
# Sequentially run closed-loop conditions for one model on one GPU.  Usage: closedloop_queue.sh <model> <gpu> <cond1> [cond2 ...]
set -u
cd "$(dirname "$0")/../.." && source scripts/probe/env.sh
MODEL=$1; GPU=$2; shift 2
for COND in "$@"; do
  echo "=== $(date) launching $MODEL / $COND on GPU $GPU"
  $PY scripts/probe/run_closedloop.py launch --model "$MODEL" --condition "$COND" --gpus "$GPU" --sims-per-gpu 6
  echo "=== $(date) finished $MODEL / $COND"
done
