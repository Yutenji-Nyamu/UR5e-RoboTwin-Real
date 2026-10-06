#!/usr/bin/env bash
# Single-node RoboDojo post-training (8 GPUs) on a training node, launched from the dev box.
#   bash scripts/robodojo/launch_t2.sh [CONFIG] [-- --set k=v ...]
# Env: NODE (default t2), MASTER_PORT (29660), SESSION (rdj_train).
# Stops the GPU keep-alive on the node first and restarts it (node-specific pid file) when training exits.
set -euo pipefail
config="${1:-configs/stage3_robodojo.yaml}"; shift || true
[[ "${1:-}" == "--" ]] && shift
extra="$*"
PROJECT=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921
node="${NODE:-t2}"
port="${MASTER_PORT:-29660}"
session="${SESSION:-rdj_train}"
out_dir="${OUT_DIR:-$(python3 - "$PROJECT/$config" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
EOF
)}"
mkdir -p "$out_dir"
stamp="$(date +%Y%m%d_%H%M%S)"
keepalive="setsid nohup /usr/bin/python3 /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py --gpus all --size 2000 \
  --pid-file /m2v_intern_v3/danglingwei/logs/${node}_after_rdj_${stamp}.pid \
  --log-file /m2v_intern_v3/danglingwei/logs/${node}_after_rdj_${stamp}.log > /dev/null 2>&1 < /dev/null &"
echo "node $node, port $port, session $session, output $out_dir"
ssh -o BatchMode=yes "$node" bash -s <<EOF
set -u
cd $PROJECT
if tmux has-session -t $session 2>/dev/null; then echo "[$node] tmux session $session already exists, abort"; exit 1; fi
if pgrep -f "^/usr/bin/python3.10 -m torch.distributed.run" >/dev/null; then echo "[$node] a torchrun is already running, abort"; exit 1; fi
pkill -f '^(/usr/bin/)?python3? /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py --gpus all' && sleep 3 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tmux new-session -d -s $session "cd $PROJECT && METIS_PYTHON=/usr/bin/python3.10 MASTER_PORT=$port \
  bash scripts/launch_stage.sh $config 1 0 127.0.0.1 ${extra:+-- $extra} \
  > $out_dir/\$(hostname -s)_${stamp}.log 2>&1; $keepalive"
sleep 2; tmux ls | grep $session && echo "[$node] launched"
EOF
echo "log: $out_dir/<host>_${stamp}.log ; tensorboard --logdir $out_dir/tensorboard"
