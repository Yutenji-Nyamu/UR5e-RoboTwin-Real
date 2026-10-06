#!/usr/bin/env bash
# Interim evaluation on the simulation node while a VLABench run is still training: wait for campaigns already running
# here, export the newest complete checkpoint of the run (bf16), and run the full 5-track campaign on all GPUs.
#   tmux new-session -d -s <name> "bash scripts/vlabench/interim_eval.sh <run dir> <config> <execute steps>"
set -uo pipefail
RUN=$1
CONFIG=$2
EXEC=${3:-16}
cd "$(dirname "$0")/../.."
LOG="$RUN/simeval/interim.log"
mkdir -p "$RUN/simeval/checkpoint"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f '^/usr/bin/python3.10 -m metiswam4d.eval.vlabench_campaign run' >/dev/null; do sleep 120; done
ckpt=$(for d in "$RUN"/checkpoints/step_*; do [[ -f $d/complete.json ]] && echo "$d"; done | sort | tail -1)
step=$(basename "$ckpt"); step=${step#step_}
MODEL="$RUN/simeval/checkpoint/model_bf16_step${step}.pt"
log "exporting $ckpt -> $MODEL"
OMP_NUM_THREADS=16 PYTHONPATH=. /usr/bin/python3.10 scripts/eval/export_bf16.py --config "$CONFIG" --checkpoint "$ckpt" \
  --out "$MODEL" >> "$LOG" 2>&1 || { log "export failed"; exit 1; }
OUT="$RUN/simeval/step${step}_e${EXEC}"
log "campaign $OUT"
bash scripts/vlabench/run_metis_simeval.sh "$OUT" "$MODEL" 0,1,2,3,4,5,6,7 8880 --execute-steps "$EXEC" --policy-seed 42
log "campaign finished"
