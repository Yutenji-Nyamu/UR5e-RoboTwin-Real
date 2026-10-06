#!/usr/bin/env bash
# Single-node 8-GPU training of the Memory / Open model.  Inside tmux on the training machine:
#   tmux new-session -d -s iw0_train "bash metiswam4d_inspired_by_internw0/scripts/run_train_local.sh <config>"
# The GPU keep-alive is stopped (exact PIDs of wangrunqi_nvml_busy.py) while training runs — the training process keeps
# the GPUs busy itself (gpu_filler) — and one instance is restored whenever this script exits.
set -uo pipefail
cd "$(dirname "$0")/../.."
config=${1:?config}
if pgrep -f '^/usr/bin/python3.10 -m torch.distributed.run' >/dev/null; then echo 'a training job already exists'; exit 1; fi
if pgrep -f 'iw0_campaign run|rdj_campaign run' >/dev/null; then echo 'stop the active campaigns before training'; exit 1; fi
out=$(/usr/bin/python3.10 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")
mkdir -p "$out"
stamp=$(date -u +%Y%m%d_%H%M%S)
host=$(hostname -s)
keep_pattern='^(/usr/(local/)?bin/)?python(3(\.1[01])?)? (/(m2v_intern_v3|ytech_milm_intern)/danglingwei/)?wangrunqi_nvml_busy.py --gpus all'
restore_keepalive() {
  if ! pgrep -f "$keep_pattern" >/dev/null; then
    setsid nohup /usr/bin/python3 /ytech_milm_intern/danglingwei/wangrunqi_nvml_busy.py --gpus all --size 2000 \
      --pid-file "/ytech_milm_intern/danglingwei/logs/${host}_after_iw0_${stamp}.pid" \
      --log-file "/ytech_milm_intern/danglingwei/logs/${host}_after_iw0_${stamp}.log" >/dev/null 2>&1 </dev/null &
    echo "keep-alive restored ($(date '+%F %T'))" >>"$out/keepalive_${stamp}.log"
  fi
}
train_pid=''
trap restore_keepalive EXIT
trap '[ -n "$train_pid" ] && kill -TERM "$train_pid" 2>/dev/null; exit 143' TERM INT
for pid in $(pgrep -f "$keep_pattern"); do
  echo "stopping keep-alive pid $pid: $(tr '\0' ' ' </proc/$pid/cmdline)" >>"$out/keepalive_${stamp}.log"
  kill -TERM "$pid"
done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD"
/usr/bin/python3.10 -m torch.distributed.run --nproc_per_node 8 --master_port 29670 \
  -m metiswam4d_inspired_by_internw0.train --config "$config" >"$out/${host}_${stamp}.log" 2>&1 &
train_pid=$!
echo "training pid=$train_pid log=$out/${host}_${stamp}.log"
status=0
wait "$train_pid" || status=$?
printf '%s\n' "$status" >"$out/train_exit_code"
exit "$status"
