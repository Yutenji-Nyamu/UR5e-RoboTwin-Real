#!/usr/bin/env bash
# EBench bridge workers (GenManip EvalClient <-> OpenWAM-style policy WebSocket servers), one process per worker
# id. Workers share policy servers (scripts/ebench/ebench_bridge.py keeps the action chunk client-side):
# worker i talks to the server on SOUTH_PORT_BASE + (i mod NUM_POLICY).
# Workers start STAGGER seconds apart: Isaac apps that start their RTX renderer at the same moment hang
# (16 simultaneous starts all stalled after "Simulation App Starting"; 45 s apart, overlapping inits are fine).
#   RUN_ID=x WORKERS="$(seq -s ' ' 0 23)" NUM_POLICY=8 [EXTRA_ARGS=--save-process] bash scripts/ebench/run_alpha_bridge.sh
set -euo pipefail
STAGGER=${STAGGER:-60}
: "${RUN_ID:?}"
WORKERS=${WORKERS:-0}
NUM_POLICY=${NUM_POLICY:-1}
SOUTH_PORT_BASE=${SOUTH_PORT_BASE:-8848}
URL=${URL:-http://127.0.0.1:8087}
CKPT_CONFIG=${CKPT_CONFIG:-/ytech_milm_intern/danglingwei/datas/EBench/OpenWAM-Alpha-Sim-EBench/config.yaml}
OUT=${OUT:-/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/ebench_eval/$RUN_ID}
LOGS=/m2v_intern_v3/danglingwei/logs/ebench/$RUN_ID
BRIDGE=$(cd "$(dirname "$0")" && pwd)/ebench_bridge.py
mkdir -p "$OUT/client_results" "$LOGS"
unset http_proxy https_proxy

cd /ytech_milm_intern/danglingwei/files/EBench_suite/OpenWAM
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
for w in $WORKERS; do
  GENMANIP_RESULT_DIR=$OUT/client_results PYTHONPATH=. /usr/local/ebench/genmanip-venv/bin/python "$BRIDGE" \
    --url "$URL" --run-id "$RUN_ID" --worker-id "$w" --south-port $((SOUTH_PORT_BASE + w % NUM_POLICY)) \
    --ckpt-config "$CKPT_CONFIG" ${EXTRA_ARGS:-} > "$LOGS/bridge_$w.log" 2>&1 &
  pids+=($!)
  sleep "$STAGGER"
done
wait
