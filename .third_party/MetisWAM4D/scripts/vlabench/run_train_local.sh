#!/usr/bin/env bash
# Single-node VLABench training on the current machine (all visible GPUs), inside tmux.  The keepalive time-slices the
# GPUs with training (16.6 s/step with it vs 4.7 s/step without on 8 x A800), so it is stopped through its own pid file
# for the run and an instance with the same arguments is started again whenever training exits.
#   tmux new-session -d -s vlab_train "bash scripts/vlabench/run_train_local.sh configs/stage3_vlabench_v1.yaml [-- --set k=v ...]"
set -euo pipefail
cd "$(dirname "$0")/../.."
config=${1:?config}
shift
if pgrep -f '^/usr/bin/python3.10 -m torch.distributed.run' >/dev/null; then
  echo 'A training job already exists'; exit 1
fi
if pgrep -f '^/usr/local/vlabench/venv/bin/python scripts/vlabench/generate_4d.py' >/dev/null && [[ ${ALLOW_WITH_GENERATION:-0} != 1 ]]; then
  echo 'VLABench generation is still running (ALLOW_WITH_GENERATION=1 to override)'; exit 1
fi
out=$(/usr/bin/python3.10 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")
mkdir -p "$out"
stamp=$(date -u +%Y%m%d_%H%M%S)
host=$(hostname -s)
keep_re='^(/usr/local/bin/)?python[0-9.]* /(m2v_intern_v3|ytech_milm_intern)/danglingwei/wangrunqi_nvml_busy.py'
keep_args=()

stop_keepalive() {
  local pid args base f
  for pid in $(pgrep -f "$keep_re"); do
    mapfile -d '' args < "/proc/$pid/cmdline"
    keep_args=("${args[@]:1}")
    base=''
    for i in "${!args[@]}"; do [[ ${args[$i]} == --pid-file ]] && base=${args[$((i + 1))]}; done
    for f in "$base" "${base%.pid}"_*.pid; do
      [[ -f "$f" && $(sed -n 's/^pid=//p' "$f") == "$pid" ]] || continue
      echo "stopping keepalive pid=$pid ($f)"
      kill -TERM "$pid"
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    done
  done
}
restore_keepalive() {
  [[ ${#keep_args[@]} -gt 0 ]] || return 0
  if ! pgrep -f "$keep_re" >/dev/null; then
    setsid nohup /usr/local/bin/python "${keep_args[@]}" >"$out/keepalive_restore_${host}_${stamp}.out" 2>&1 </dev/null &
    echo "keepalive restored pid=$! args=${keep_args[*]}" | tee -a "$out/keepalive_restore_${host}_${stamp}.out"
  fi
}
train_pid=''
trap restore_keepalive EXIT
trap '[[ -z "$train_pid" ]] || kill -TERM "$train_pid" 2>/dev/null || true; exit 143' TERM INT HUP
stop_keepalive
if [[ ${STOP_MPS:-0} == 1 ]] && pgrep -f '^nvidia-cuda-mps-control' >/dev/null; then
  # an evaluation left the CUDA MPS daemon on; training runs as plain CUDA processes
  echo quit | nvidia-cuda-mps-control
  for _ in $(seq 30); do pgrep -f '^nvidia-cuda-mps' >/dev/null || break; sleep 1; done
  echo "CUDA MPS daemon stopped"
fi
export METIS_PYTHON=/usr/bin/python3.10 MASTER_PORT=${MASTER_PORT:-29671}
export GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/launch_stage.sh "$config" 1 0 127.0.0.1 "$@" >"$out/${host}_${stamp}.log" 2>&1 &
train_pid=$!
echo "training pid=$train_pid gpus=$GPUS_PER_NODE log=$out/${host}_${stamp}.log"
status=0
wait "$train_pid" || status=$?
printf '%s\n' "$status" >"$out/train_exit_code"
exit "$status"
