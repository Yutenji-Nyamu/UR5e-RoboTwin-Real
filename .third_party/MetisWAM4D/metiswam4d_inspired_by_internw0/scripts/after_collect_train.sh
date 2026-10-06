#!/usr/bin/env bash
# Hand-over on the jump host: wait for the teacher collection managers on every collection host to exit, convert the
# remaining recordings, assemble the training root, then start the 8-GPU training on the training host.
#   tmux new-session -d -s iw0_chain "bash metiswam4d_inspired_by_internw0/scripts/after_collect_train.sh <train host> <collection hosts...>"
set -uo pipefail
TRAIN_HOST=$1
shift
COLLECT_HOSTS=("$@")
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT=/ytech_milm_intern/danglingwei/datas/IW0_MemOpen4D/dataset_v1
CONFIG=metiswam4d_inspired_by_internw0/configs/iw0_memopen_v1.yaml
LOG=/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/iw0_memopen_v1/chain.log
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
cd "$PROJECT" || exit 2

active() {
  local h n
  for h in "${COLLECT_HOSTS[@]}"; do
    n=$(ssh -o ConnectTimeout=10 "$h" "pgrep -fc '^/usr/bin/python3.10 -m metiswam4d_inspired_by_internw0.eval.iw0_campaign run'" 2>/dev/null)
    [[ ${n:-0} -gt 0 ]] && { echo "$h:$n"; return 0; }
  done
  return 1
}
log "waiting for the collection managers on ${COLLECT_HOSTS[*]}"
while state=$(active); do sleep 120; done
log "collection finished; converting"
tmux kill-session -t iw0_convert 2>/dev/null
while pgrep -f '^/usr/bin/python3.10 metiswam4d_inspired_by_internw0/data_prep/build_gt4d.py' >/dev/null; do sleep 10; done
export PYTHONPATH="$PROJECT:/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824"
export VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/nvidia_icd.json
REC=/ytech_milm_intern/danglingwei/datas/IW0_MemOpen4D/records_t1
for spec in "seed1 0" "seed2 10000" "seed0 20000" "seed1b 0" "seed2b 10000" "seed0b 20000" \
            "seed1r 0" "seed2r 10000" "seed0r 20000"; do
  set -- $spec
  [ -d "$REC/$1" ] || continue
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 /usr/bin/python3.10 metiswam4d_inspired_by_internw0/data_prep/build_gt4d.py build \
    --record "$REC/$1" --out "$ROOT" --tag t1 --workers 12 --episode-offset "$2" 2>&1 | grep -v -i warn | grep -v -e '"status": "exists"' -e 'episode collision' >> "$LOG"
done
log "assembling $ROOT"
PYTHONPATH="$PROJECT" OMP_NUM_THREADS=8 /usr/bin/python3.10 metiswam4d_inspired_by_internw0/data_prep/assemble_dataset.py \
  --root "$ROOT" --tags t1 --repeat 2 --task-repeat general_pickup=4,stack_blocks_by_language=4 --teacher-val 3 \
  --device cuda:0 2>&1 | grep -v -i warn >> "$LOG" || { log "assembly failed"; exit 1; }
log "starting training on $TRAIN_HOST"
ssh "$TRAIN_HOST" "cd $PROJECT && tmux new-session -d -s iw0_train 'bash metiswam4d_inspired_by_internw0/scripts/run_train_local.sh $CONFIG'" \
  && log "tmux iw0_train started on $TRAIN_HOST" || log "training launch failed"
