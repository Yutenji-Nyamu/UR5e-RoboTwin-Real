#!/usr/bin/env bash
# Build the next focus-task self-play dataset: SAPIEN geometry + FK Track for every new recording directory, then
# finalize with the nine focus tasks' official demos and the unique rollouts of the previous dataset (repeat 3).
#   bash scripts/robodojo/build_focus_dataset.sh <out> <tag> <previous dataset> <record dir>...
# Runs on one GPU (SAPIEN needs Vulkan); logs to <out>/_logs/.  Recording directories get disjoint episode offsets
# (0, 10000, 20000, ...) so the same task/layout numbers from different seeds never collide.
set -euo pipefail
out=$1; tag=$2; previous=$3; shift 3
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:/m2v_intern_v3/danglingwei/codes/JanusTrack4d_260824
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
tasks=match_and_pick_from_conveyor,cover_blocks,build_tower,pour_balls_into_vase,insert_tubes,play_tic_tac_toe,stack_blocks,pour_liquid_into_cup,fold_clothes
mkdir -p "$out/_logs"
log="$out/_logs/build_${tag}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
echo "build $tag -> $out (previous $previous) records: $*"
offset=0
for record in "$@"; do
  echo "== build $record (offset $offset)"
  /usr/bin/python3.10 scripts/data_prep/robodojo/rdj_rollout_dataset.py build --record "$record" --out "$out" --tag "$tag" \
    --workers 8 --tasks "$tasks" --episode-offset "$offset"
  offset=$((offset + 10000))
done
echo "== finalize"
/usr/bin/python3.10 scripts/data_prep/robodojo/rdj_rollout_dataset.py finalize --out "$out" --tag "$tag" --repeat 3 \
  --tasks "$tasks" --include-dataset "$previous"
echo "== verify"
/usr/bin/python3.10 scripts/data_prep/robodojo/verify_focus_dataset.py --root "$out" --repeat 3
echo "build $tag done"
