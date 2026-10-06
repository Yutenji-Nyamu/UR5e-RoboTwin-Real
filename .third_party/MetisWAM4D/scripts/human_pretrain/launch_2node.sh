#!/usr/bin/env bash
# Two-node stage-1 human pretraining (NODES[0] = rank 0 / master; 8 GPUs each, HSDP: shard in node, replicate across).
# Run from the dev box.  Each node runs torchrun inside tmux session "human_pretrain"; logs go next to the checkpoints.
#   NODES="tX tY" bash scripts/human_pretrain/launch_2node.sh [CONFIG] [-- --set k=v ...]
# Env: NODES (first = master), MASTER_ADDR (default: first IP of the master), MASTER_PORT (29661), OUT_DIR,
#      PROJECT (source tree the nodes run from; point it at a frozen copy so later edits cannot reach the run).
# Nodes need /usr/bin/python3.10 with smplx / pytorch3d (MANO meshes in the workers, GPU rasterisation), diffusers, transformers.
set -euo pipefail
config="${1:-configs/stage1_pretrain_human.yaml}"; shift || true
[[ "${1:-}" == "--" ]] && shift
extra="$*"
PROJECT="${PROJECT:-/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921}"
read -r -a nodes <<< "${NODES:?set NODES=\"tX tY\"}"
master="${nodes[0]}"
master_addr="${MASTER_ADDR:-$(ssh -o BatchMode=yes "$master" 'hostname -I | cut -d" " -f1')}"
port="${MASTER_PORT:-29661}"
out_dir="${OUT_DIR:-$(python3 - "$PROJECT/$config" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
EOF
)}"
mkdir -p "$out_dir"
echo "master $master ($master_addr:$port), nodes: ${nodes[*]}, output: $out_dir"
for rank in "${!nodes[@]}"; do
  node="${nodes[$rank]}"
  ssh -o BatchMode=yes "$node" bash -s <<EOF
set -u
cd $PROJECT
/usr/bin/python3.10 -c "import smplx, pytorch3d, diffusers, transformers, av" || { echo "[$node] missing python3.10 deps"; exit 1; }
# The keep-alive guard (~420 MiB/GPU, shown as "[Not Found]" inside the pod) stays up; anything above 1 GiB is another job.
busy=\$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '\$1 > 1024' | wc -l)
if [[ "\$busy" -gt 0 ]]; then echo "[$node] other GPU processes running:"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; exit 1; fi
tmux kill-session -t human_pretrain 2>/dev/null || true
tmux new-session -d -s human_pretrain "cd $PROJECT && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True METIS_PYTHON=/usr/bin/python3.10 MASTER_PORT=$port \
  bash scripts/launch_stage.sh $config ${#nodes[@]} $rank $master_addr ${extra:+-- $extra} \
  > $out_dir/\$(hostname -s)_rank$rank.log 2>&1"
sleep 2; tmux ls | grep human_pretrain && echo "[$node] rank $rank launched"
EOF
done
echo "logs: $out_dir/<host>_rank{0,1}.log ; tensorboard --logdir $out_dir/tensorboard"
