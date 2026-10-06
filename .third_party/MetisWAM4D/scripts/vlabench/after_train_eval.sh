#!/usr/bin/env bash
# Evaluation chain on the simulation node for a VLABench training run on any node: wait for the run's train_exit_code
# (written by run_train_local.sh when training ends), export the mean of its last N complete checkpoints (bf16), wait
# for any campaign already running here, then run the full 5-track campaign on all GPUs of this node.
#   tmux new-session -d -s <name> "bash scripts/vlabench/after_train_eval.sh <run dir> <config> <n ckpts> <execute steps>"
set -uo pipefail
RUN=$1
CONFIG=$2
N=${3:-4}
EXEC=${4:-16}
cd "$(dirname "$0")/../.."
LOG="$RUN/simeval/chain.log"
mkdir -p "$RUN/simeval/checkpoint"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "waiting for $RUN/train_exit_code"
while [[ ! -f "$RUN/train_exit_code" ]]; do sleep 120; done
code=$(cat "$RUN/train_exit_code")
log "training exited with $code"
[[ $code == 0 ]] || { log "not chaining the evaluation"; exit 1; }
mapfile -t ckpts < <(for d in "$RUN"/checkpoints/step_*; do [[ -f $d/complete.json ]] && echo "$d"; done | sort | tail -n "$N")
first=$(basename "${ckpts[0]}"); last=$(basename "${ckpts[-1]}")
MODEL="$RUN/simeval/checkpoint/model_bf16_avg${#ckpts[@]}_${first#step_}_${last#step_}.pt"
log "averaging ${ckpts[*]} -> $MODEL"
OMP_NUM_THREADS=16 PYTHONPATH=. /usr/bin/python3.10 scripts/eval/export_bf16.py --config "$CONFIG" \
  --checkpoint "$(IFS=,; echo "${ckpts[*]}")" --out "$MODEL" >> "$LOG" 2>&1 || { log "export failed"; exit 1; }
while pgrep -f '^/usr/bin/python3.10 -m metiswam4d.eval.vlabench_campaign run' >/dev/null; do sleep 120; done
OUT="$RUN/simeval/avg${#ckpts[@]}_e${EXEC}"
log "campaign $OUT"
bash scripts/vlabench/run_metis_simeval.sh "$OUT" "$MODEL" 0,1,2,3,4,5,6,7 8880 --execute-steps "$EXEC" --policy-seed 42
log "campaign finished"
