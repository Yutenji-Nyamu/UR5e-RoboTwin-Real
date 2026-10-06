#!/usr/bin/env bash
# Two-node RT2 training (NODES[0] = rank 0 / master, NODES[1] = rank 1; 8 GPUs each, HSDP: shard in node, replicate across).
# Run from the dev box.  Each node runs torchrun inside tmux session "rt2_train"; logs go next to the checkpoints.
#   bash scripts/launch_rt2_2node.sh [CONFIG] [-- --set k=v ...]
# Env: NODES="t3 t4" (first = master), MASTER_ADDR (default: eth0 IP of the master), MASTER_PORT (29655).
set -euo pipefail
config="${1:-configs/rt2_direct.yaml}"; shift || true
[[ "${1:-}" == "--" ]] && shift
extra="$*"
PROJECT=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921
read -r -a nodes <<< "${NODES:-t3 t4}"
master="${nodes[0]}"
master_addr="${MASTER_ADDR:-$(ssh -o BatchMode=yes "$master" 'hostname -I | cut -d" " -f1')}"
port="${MASTER_PORT:-29655}"
out_dir="${OUT_DIR:-$(python3 - "$PROJECT/$config" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
EOF
)}"   # OUT_DIR: log location when output_dir is overridden via --set
mkdir -p "$out_dir"
echo "master $master ($master_addr:$port), nodes: ${nodes[*]}, output: $out_dir"
for rank in "${!nodes[@]}"; do
  node="${nodes[$rank]}"
  ssh -o BatchMode=yes "$node" bash -s <<EOF
set -u
cd $PROJECT
if pgrep -f "^/usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py run" >/dev/null; then echo "[$node] RT2 shards still running, abort"; exit 1; fi
tmux kill-session -t gpu_filler 2>/dev/null || true
tmux kill-session -t rt2_train 2>/dev/null || true
tmux new-session -d -s rt2_train "cd $PROJECT && METIS_PYTHON=/usr/bin/python3.10 MASTER_PORT=$port \
  bash scripts/launch_stage.sh $config ${#nodes[@]} $rank $master_addr ${extra:+-- $extra} \
  > $out_dir/\$(hostname -s)_rank$rank.log 2>&1"
sleep 2; tmux ls | grep rt2_train && echo "[$node] rank $rank launched"
EOF
done
echo "logs: $out_dir/<host>_rank{0,1}.log ; tensorboard --logdir $out_dir/tensorboard"
