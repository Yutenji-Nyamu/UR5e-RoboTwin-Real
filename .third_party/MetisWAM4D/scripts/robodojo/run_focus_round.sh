#!/usr/bin/env bash
# One focus-task self-play round on t2, unattended: wait for the running collection to finish, build the dataset,
# train, then export the tail-3 average and evaluate the 12 focus configs (hide mode only, 120 episodes).
#   tmux new-session -d -s rdj_round "bash scripts/robodojo/run_focus_round.sh <config> <dataset_out> <tag> \
#       <previous_dataset> <previous_focus_dirs comma-separated> <record_dir>..."
# Every stage appends to <run>/pipeline.log; the keep-alive is restored by run_focus_train_local.sh when training exits.
set -euo pipefail
config=$1; dataset=$2; tag=$3; previous_dataset=$4; previous_focus=$5; shift 5
cd "$(dirname "$0")/../.."
run=$(/usr/bin/python3.10 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")
mkdir -p "$run"
log="$run/pipeline.log"
stamp() { date '+%F %T'; }
echo "$(stamp) round start: config=$config dataset=$dataset records=$*" | tee -a "$log"

# 1. wait for the collection on this host (campaign managers / policy servers / Isaac clients) to finish
while pgrep -f '^/usr/bin/python3.10 scripts/robodojo/focus_collect.py' >/dev/null || \
      pgrep -f '^/usr/bin/python3.10 -m metiswam4d.eval.rdj_(campaign|policy)' >/dev/null || \
      pgrep -f '^/usr/local/robodojo-python-runtimes/.*rdj_client' >/dev/null; do
  sleep 60
done
echo "$(stamp) collection finished on this host" | tee -a "$log"

# 2. dataset (GPU 0; the keep-alive stays up)
if [[ ! -f "$dataset/index.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=0 bash scripts/robodojo/build_focus_dataset.sh "$dataset" "$tag" "$previous_dataset" "$@" \
    2>&1 | tail -n 5 | tee -a "$log" || true
  [[ -f "$dataset/index.jsonl" && -f "$dataset/validation.json" ]] || { echo "$(stamp) dataset build failed" | tee -a "$log"; exit 1; }
fi
echo "$(stamp) dataset ready: $(wc -l <"$dataset/index.jsonl") index rows" | tee -a "$log"

# 3. train (stops the keep-alive right before torchrun, restores it on exit)
unset CUDA_VISIBLE_DEVICES
rm -f "$run/train_exit_code"
bash scripts/robodojo/run_focus_train_local.sh "$config" 2>&1 | tee -a "$log" || true
code=$(cat "$run/train_exit_code" 2>/dev/null || echo launch_failed)
[[ "$code" == 0 ]] || { echo "$(stamp) training exited with $code" | tee -a "$log"; exit "$code"; }

# 4. export + evaluate + report
prev_args=()
IFS=',' read -ra prev <<<"$previous_focus"
for p in "${prev[@]}"; do [[ -n "$p" ]] && prev_args+=(--previous-focus "$p"); done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /usr/bin/python3.10 scripts/robodojo/focus_after_train.py --config "$config" \
  --port 33880 --variants '{"hide": {"hide_track": true}}' "${prev_args[@]}" 2>&1 | tee -a "$log"
echo "$(stamp) round complete" | tee -a "$log"
