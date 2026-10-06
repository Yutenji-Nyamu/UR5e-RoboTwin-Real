#!/usr/bin/env bash
# Short two-node throughput benchmarks of configs/rt2_direct.yaml variants (no checkpoints, no vis).
# Each variant runs STEPS optimizer steps on t3+t4 and reports the per-phase step time from the log.
#   bash scripts/bench_rt2_variants.sh
set -uo pipefail
PROJECT=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921
BENCH=/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/rt2_bench
STEPS="${STEPS:-30}"
common="training.checkpoint_every=0 training.checkpoint_every_minutes=0 training.visualize=false training.log_every=10 init.resume_from=null"
declare -A variants=(
  [bs4_acc2_block]="data.batch_size=4 training.grad_accumulation=2"
  [bs8_acc1_block]="data.batch_size=8 training.grad_accumulation=1"
  [bs8_acc1_root_keep]="data.batch_size=8 training.grad_accumulation=1 training.fsdp_granularity=root training.fsdp_reshard_after_forward=false"
  [bs16_acc1_block_ckpt]="data.batch_size=16 training.grad_accumulation=1 model.gradient_checkpointing=true"
)
order="${VARIANTS:-bs8_acc1_block bs8_acc1_root_keep bs16_acc1_block_ckpt bs4_acc2_block}"
for name in $order; do
  out="$BENCH/$name"; mkdir -p "$out"
  echo "=== $name: ${variants[$name]}"
  cd "$PROJECT" && OUT_DIR="$out" MASTER_PORT=29660 bash scripts/launch_rt2_2node.sh configs/rt2_direct.yaml -- \
    --set output_dir="$out" $common ${variants[$name]} --stop-after-steps "$STEPS" >/dev/null
  # wait for both tmux sessions to exit (max 20 min)
  for _ in $(seq 1 120); do
    sleep 10
    alive=0
    for node in t3 t4; do ssh -o BatchMode=yes "$node" 'tmux has-session -t rt2_train 2>/dev/null' && alive=1; done
    [[ "$alive" == "0" ]] && break
  done
  grep -E "^\[step|out of memory|Error" "$out"/*_rank0.log | tail -n 3 | cut -c1-260
done
