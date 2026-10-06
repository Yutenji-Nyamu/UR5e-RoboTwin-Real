#!/usr/bin/env bash
# Stop this host's RT2 campaign: the tmux session and the whole process tree of the manager recorded in
# <out>/logs/manager_<host>.json (model workers and their spawned simulation processes).
#   bash scripts/eval/stop_rt2_simeval.sh <out> <tmux session>
set -uo pipefail
OUT=$1
SESSION=$2
host=$(hostname | cut -d. -f1)
manager=$(/usr/bin/python3.10 -c "import json,sys; print(json.load(open(sys.argv[1]))['pid'])" "$OUT/logs/manager_${host}.json")

tree() {
  echo "$1"
  for child in $(pgrep -P "$1"); do tree "$child"; done
}
pids=$(tree "$manager")
tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed tmux $SESSION"
for pid in $pids; do kill "$pid" 2>/dev/null; done
sleep 10
for pid in $pids; do kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; done
echo "stopped manager $manager and $(echo "$pids" | wc -w) processes"
