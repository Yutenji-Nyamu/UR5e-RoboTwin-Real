#!/usr/bin/env bash
# Convert teacher recordings to ground-truth 4D episodes while the collection runs (every ROUND seconds; converted
# episodes are skipped).  Inside tmux on the collection host:
#   tmux new-session -d -s iw0_convert "bash metiswam4d_inspired_by_internw0/scripts/convert_loop.sh"
set -uo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
REC=${REC:-/ytech_milm_intern/danglingwei/datas/IW0_MemOpen4D/records_t1}
OUT=${OUT:-/ytech_milm_intern/danglingwei/datas/IW0_MemOpen4D/dataset_v1}
TAG=${TAG:-t1}
ROUND=${ROUND:-1200}
GPU=${GPU:-7}
cd "$PROJECT" || exit 2
export PYTHONPATH="$PROJECT:/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824"
export VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while true; do
  # *b = second-half campaigns of the same seeds: same numbering (episode = offset + layout id), duplicates collide;
  # *r = Open-task retries of the layouts without a success, same numbering
  for spec in "seed1 0" "seed2 10000" "seed0 20000" "seed1b 0" "seed2b 10000" "seed0b 20000" \
              "seed1r 0" "seed2r 10000" "seed0r 20000"; do
    set -- $spec
    [ -d "$REC/$1" ] || continue
    nice -n 10 /usr/bin/python3.10 metiswam4d_inspired_by_internw0/data_prep/build_gt4d.py build --record "$REC/$1" \
      --out "$OUT" --tag "$TAG" --workers 6 --episode-offset "$2" 2>&1 | grep -v -i warn | grep -v '"status": "exists"'
  done
  echo "[convert_loop] round done $(date '+%F %T')"
  sleep "$ROUND"
done
