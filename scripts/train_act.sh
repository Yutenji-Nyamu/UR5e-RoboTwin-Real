#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 TASK_NAME TASK_CONFIG EPISODES SEED GPU_ID" >&2
  exit 2
fi

TASK_NAME="$1"
TASK_CONFIG="$2"
EPISODES="$3"
SEED="$4"
GPU_ID="$5"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACT_ROOT="${PROJECT_ROOT}/.third_party/RoboTwin/policy/ACT"

if [[ ! -f "${ACT_ROOT}/imitate_episodes.py" ]]; then
  echo "RoboTwin ACT is missing; run scripts/bootstrap_robotwin.sh first" >&2
  exit 1
fi

cd "${ACT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python imitate_episodes.py \
  --task_name "sim-${TASK_NAME}-${TASK_CONFIG}-${EPISODES}" \
  --ckpt_dir "./act_ckpt/act-${TASK_NAME}/${TASK_CONFIG}-${EPISODES}" \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 6000 \
  --lr 1e-5 \
  --save_freq 2000 \
  --state_dim 14 \
  --seed "${SEED}"
