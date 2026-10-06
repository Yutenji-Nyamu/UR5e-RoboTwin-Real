#!/usr/bin/env bash
# Run on any node: once every clean episode is complete, rebuild index.jsonl and the frozen norm /
# depth statistics from the clean set (the randomized shards keep running from the same launch).
set -uo pipefail
PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT"
PY=/usr/bin/python3.10
while true; do
  done_clean=$($PY scripts/data_prep/rt2/rt2_track4d.py status --variants demo_clean_4d | $PY -c "import json,sys; print(json.load(sys.stdin)['complete'])")
  echo "$(date '+%F %T') clean complete: $done_clean / 2500"
  [[ "$done_clean" -ge 2500 ]] && break
  sleep 300
done
$PY scripts/data_prep/rt2/rt2_track4d.py index --variants demo_clean_4d
$PY scripts/data_prep/rt2/rt2_track4d.py norm --sample-per-task 6 --variants demo_clean_4d
echo "$(date '+%F %T') index + norm done (clean)"
