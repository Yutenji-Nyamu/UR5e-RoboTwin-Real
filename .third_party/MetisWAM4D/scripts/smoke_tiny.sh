#!/usr/bin/env bash
# Tiny synthetic smoke: unit tests, a few CPU training steps with checkpoint + resume,
# and (if GPUs are visible) a 2-process FSDP run of the same config.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$METIS_PROJECT"
"$METIS_PYTHON" -m pytest -q tests
out=/tmp/metiswam4d_smoke_$$
CUDA_VISIBLE_DEVICES="" "$METIS_PYTHON" -m metiswam4d.train.train --config configs/smoke_tiny.yaml --set output_dir="$out"
CUDA_VISIBLE_DEVICES="" "$METIS_PYTHON" -m metiswam4d.train.train --config configs/smoke_tiny.yaml --set output_dir="$out" init.resume_from=auto training.max_steps=8
if command -v nvidia-smi >/dev/null && [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ge 2 ]]; then
  "$METIS_PYTHON" -m torch.distributed.run --nproc_per_node 2 --master_port "${MASTER_PORT:-29652}" \
    -m metiswam4d.train.train --config configs/smoke_tiny.yaml \
    --set output_dir="${out}_fsdp" training.fsdp=true training.dtype=bf16 training.max_steps=4 training.checkpoint_every=2
fi
rm -rf "$out" "${out}_fsdp"
echo "smoke ok"
