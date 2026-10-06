#!/usr/bin/env bash
# Dev-box chain: wait until after_all_index.sh has written the final index, then start the two-node RT2
# training (scripts/launch_rt2_2node.sh: t3 rank 0 + t4 rank 1).  Only our own gpu_filler sessions are
# stopped on the nodes; the shared keep-alive process is left alone (it coexists with training).
#   bash scripts/data_prep/rt2/launch_training_after_index.sh
set -uo pipefail
WATCH=/m2v_intern_v3/danglingwei/logs/rt2_after_all_index.out
PROJECT=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921
while ! grep -q "final index done" "$WATCH" 2>/dev/null; do
  echo "$(date '+%F %T') waiting for final index ($(tail -n 1 "$WATCH" 2>/dev/null))"
  sleep 300
done
echo "$(date '+%F %T') index ready; launching two-node training"
cd "$PROJECT" && NODES="${NODES:-t3 t4}" bash scripts/launch_rt2_2node.sh configs/rt2_direct.yaml
echo "$(date '+%F %T') launched"
