#!/usr/bin/env bash
# Run inside tmux on t2. Restore one keepalive instance whenever training exits.
set -euo pipefail
cd "$(dirname "$0")/../.."
config=${1:-configs/stage3_robodojo_v5_focus.yaml}
[[ $(hostname -s) == a800bcctest0080-bd ]] || { echo 'This run is authorized only on t2'; exit 1; }
if pgrep -f '^/usr/bin/python3.10 -m torch.distributed.run' >/dev/null; then
  echo 'A training job already exists'; exit 1
fi
if pgrep -f '^/usr/bin/python3.10 -m metiswam4d.eval.rdj_(campaign|policy)' >/dev/null; then
  echo 'Stop the active campaign before training'; exit 1
fi
out=$(/usr/bin/python3.10 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")
mkdir -p "$out"
stamp=$(date -u +%Y%m%d_%H%M%S)
keep_pattern='^(/usr/bin/)?python(3(\.10)?)? (/m2v_intern_v3/danglingwei/)?wangrunqi_nvml_busy.py --gpus all'
train_pid=''
restore_keepalive() {
  if ! pgrep -f "$keep_pattern" >/dev/null; then
    setsid nohup /usr/bin/python3 /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py --gpus all --size 2000 \
      --pid-file "/m2v_intern_v3/danglingwei/logs/t2_after_rdj_${stamp}.pid" \
      --log-file "/m2v_intern_v3/danglingwei/logs/t2_after_rdj_${stamp}.log" >/dev/null 2>&1 </dev/null &
  fi
}
stop_training() { [[ -z "$train_pid" ]] || kill -TERM "$train_pid" 2>/dev/null || true; }
trap restore_keepalive EXIT
trap 'stop_training; exit 143' TERM INT
export METIS_PYTHON=/usr/bin/python3.10 MASTER_PORT=29660 GPUS_PER_NODE=8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pkill -f "$keep_pattern" || true
bash scripts/launch_stage.sh "$config" 1 0 127.0.0.1 >"$out/t2_${stamp}.log" 2>&1 &
train_pid=$!
echo "training pid=$train_pid log=$out/t2_${stamp}.log"
status=0
wait "$train_pid" || status=$?
printf '%s\n' "$status" >"$out/train_exit_code"
exit "$status"
