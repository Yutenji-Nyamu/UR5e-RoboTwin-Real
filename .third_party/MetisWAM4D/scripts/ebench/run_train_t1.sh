#!/usr/bin/env bash
# Single-node EBench training on t1, inside tmux. Stops the keepalive for the run and restores one instance
# whenever training exits (any GPU process left next to it would be a plain CUDA process vs MPS: see the EBench doc).
#   tmux new-session -d -s ebench_train "bash scripts/ebench/run_train_t1.sh configs/stage3_ebench_v1.yaml"
set -euo pipefail
cd "$(dirname "$0")/../.."
config=${1:?config}
[[ $(hostname -s) == a800bcctest0125-bd ]] || { echo 'This wrapper is for t1'; exit 1; }
if pgrep -f '^/usr/bin/python3.10 -m torch.distributed.run' >/dev/null; then
  echo 'A training job already exists'; exit 1
fi
if pgrep -f '^ray::IsaacWorker|^/usr/bin/python3.10 scripts/deploy.py' >/dev/null; then
  echo 'Stop the EBench evaluation (Isaac workers / policy servers) before training'; exit 1
fi
out=$(/usr/bin/python3.10 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")
mkdir -p "$out"
stamp=$(date -u +%Y%m%d_%H%M%S)
keep_pattern='^(/usr/bin/)?python(3(\.10)?)? (/m2v_intern_v3/danglingwei/)?wangrunqi_nvml_busy.py --gpus all'
train_pid=''
restore_keepalive() {
  if ! pgrep -f "$keep_pattern" >/dev/null; then
    (cd /m2v_intern_v3/danglingwei && setsid nohup python wangrunqi_nvml_busy.py --gpus all --size 2000 \
      --pid-file "logs/t1_after_ebench_${stamp}.pid" --log-file "logs/t1_after_ebench_${stamp}.log" \
      >/dev/null 2>&1 </dev/null &)
  fi
}
stop_training() { [[ -z "$train_pid" ]] || kill -TERM "$train_pid" 2>/dev/null || true; }
trap restore_keepalive EXIT
trap 'stop_training; exit 143' TERM INT
export METIS_PYTHON=/usr/bin/python3.10 MASTER_PORT=29670 GPUS_PER_NODE=8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/ytech_milm_intern/danglingwei/files/EBench_suite/OpenWAM${PYTHONPATH:+:$PYTHONPATH}
pkill -f "$keep_pattern" || true
bash scripts/launch_stage.sh "$config" 1 0 127.0.0.1 >"$out/t1_${stamp}.log" 2>&1 &
train_pid=$!
echo "training pid=$train_pid log=$out/t1_${stamp}.log"
status=0
wait "$train_pid" || status=$?
printf '%s\n' "$status" >"$out/train_exit_code"
exit "$status"
