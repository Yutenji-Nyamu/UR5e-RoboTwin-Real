#!/usr/bin/env bash
# Launch one training stage with torchrun.
#   bash scripts/launch_stage.sh configs/stage1_pretrain_human.yaml [NNODES NODE_RANK MASTER_ADDR] [-- extra --set overrides]
# Env: GPUS_PER_NODE (default: all visible), MASTER_PORT (default 29651).
set -euo pipefail
source "$(dirname "$0")/env.sh"
config="${1:?usage: launch_stage.sh CONFIG [NNODES NODE_RANK MASTER_ADDR] [-- --set k=v ...]}"
shift
nnodes="${1:-1}"; [[ $# -gt 0 ]] && shift || true
node_rank="${1:-0}"; [[ $# -gt 0 ]] && shift || true
master_addr="${1:-127.0.0.1}"; [[ $# -gt 0 ]] && shift || true
[[ "${1:-}" == "--" ]] && shift
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
  GPUS_PER_NODE="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
fi
cd "$METIS_PROJECT"
exec "$METIS_PYTHON" -m torch.distributed.run \
  --nnodes "$nnodes" --node_rank "$node_rank" --nproc_per_node "$GPUS_PER_NODE" \
  --master_addr "$master_addr" --master_port "${MASTER_PORT:-29651}" \
  -m metiswam4d.train.train --config "$config" "$@"
