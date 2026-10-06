#!/usr/bin/env bash
# Closed-loop evaluation of Memory / Open checkpoints on this host (protocol layouts: env seed 0, layouts 0-9, the 14
# Memory + Open task configs): export the mean of the given DCP checkpoints (bf16), prepare a mem-backend campaign,
# run it on the given GPUs.  Inside tmux on a BCC node without CUDA MPS (Isaac):
#   tmux new-session -d -s iw0_eval "bash metiswam4d_inspired_by_internw0/scripts/eval_ckpt.sh <out> <gpus> <execute steps> <ckpt dir>[,<ckpt dir>...]"
set -uo pipefail
OUT=$1
GPUS=$2
EXEC=$3
CKPTS=$4
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG=$PROJECT/metiswam4d_inspired_by_internw0/configs/iw0_memopen_v1.yaml
MODEL=$OUT/model_bf16.pt
cd "$PROJECT" || exit 2
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/eval.log"; }
if [[ ! -f $MODEL ]]; then
  log "exporting $CKPTS"
  OMP_NUM_THREADS=16 PYTHONPATH=. /usr/bin/python3.10 scripts/eval/export_bf16.py --builder iw0 --config "$CONFIG" \
    --checkpoint "$CKPTS" --out "$MODEL" >> "$OUT/eval.log" 2>&1 || { log "export failed"; exit 1; }
fi
TASKS=(cover_blocks match_and_pick_from_conveyor swap_blocks swap_T press_by_number imitate_sorting_sequence
       align_blocks general_pickup stack_blocks_by_language solve_equation classify_objects_by_language
       pick_from_conveyor_by_image store_tools_in_toolbox pour_by_language)
if [[ ! -f $OUT/campaign/campaign.json ]]; then
  only=(); for t in "${TASKS[@]}"; do only+=(--only "$t"); done
  (source /m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo/scripts/activate_robodojo.sh >/dev/null && \
   cd "$PROJECT" && PYTHONPATH="$PROJECT" /usr/bin/python3.10 -m metiswam4d_inspired_by_internw0.eval.iw0_campaign prepare \
     --output "$OUT/campaign" --backend mem --config "$CONFIG" --model-file "$MODEL" --execute-steps "$EXEC" \
     --policy-seed 42 "${only[@]}" --env-seed 0 --episodes-per-round 2 --video-episodes 2 --clients-per-gpu 3 \
     --info "mem_e$EXEC") >> "$OUT/eval.log" 2>&1 || { log "prepare failed"; exit 1; }
fi
log "campaign $OUT/campaign on GPUs $GPUS"
bash metiswam4d_inspired_by_internw0/scripts/run_iw0_campaign.sh "$OUT/campaign" "$GPUS" >> "$OUT/eval.log" 2>&1
log "campaign finished"
