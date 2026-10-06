#!/usr/bin/env bash
# One worker per GPU on t3. Re-runnable: finished task dirs are skipped, in-flight ones are locked.
#   ssh t3 'bash /m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/data_prep/launch_ig10k_depth_masks.sh'
#
# MUST use /usr/bin/python3.10: the vendored pycocotools _mask ext is cpython-310 and DA3 needs
# moviepy.editor, which only exists in the moviepy 1.x installed for 3.10.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/usr/bin/python3.10
OUT_ROOT="${OUT_ROOT:-/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/human}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
STAGES="${STAGES:-depth,masks}"
mkdir -p "$OUT_ROOT/_logs"
cd "$PROJ"
# t3's GPUs are shared with another tenant that holds ~45 GB, so keep fragmentation down
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" scripts/data_prep/ig10k_human_depth_masks.py \
      --out-root "$OUT_ROOT" run --stages "$STAGES" --shuffle \
      >> "$OUT_ROOT/_logs/driver_gpu${g}.out" 2>&1 < /dev/null &
  sleep 1
done
sleep 3
echo "launched on $(hostname -s); workers: $(pgrep -fc ig10k_human_depth_masks.py)"
