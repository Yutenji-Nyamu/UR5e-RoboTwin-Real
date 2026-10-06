#!/usr/bin/env bash
# One host of an InternW0-Delta RoboDojo campaign (evaluation or teacher collection).  Inside tmux:
#   tmux new-session -d -s iw0eval "bash metiswam4d_inspired_by_internw0/scripts/run_iw0_campaign.sh <out> <gpus> [clients] [keepalive pid]"
# Isaac clients run in the RoboDojo runtime, InternW0 servers in /ytech_milm_intern/danglingwei/envs/internw0-delta;
# the manager's watchdog keeps the GPU keep-alive up.  A failed manager is restarted after 60 s (up to 20 times).
set -uo pipefail
OUT=$1
GPUS=$2
CLIENTS=${3:-}
KEEP_PID=${4:-}
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"

# shellcheck source=/dev/null
source /m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo/scripts/activate_robodojo.sh || exit 2
cd "$PROJECT" || exit 2
export PYTHONPATH="$PROJECT"
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

args=(--output "$OUT" --gpus "$GPUS")
[ -n "$CLIENTS" ] && args+=(--clients "$CLIENTS")
[ -n "$KEEP_PID" ] && args+=(--keepalive-pid-file "$KEEP_PID")
[ -n "${IW0_BASE_PORT:-}" ] && args+=(--base-port "$IW0_BASE_PORT")
[ -n "${IW0_STALL_SECONDS:-}" ] && args+=(--stall-seconds "$IW0_STALL_SECONDS")

status=1
for attempt in $(seq 1 20); do
  echo "iw0 campaign manager attempt $attempt at $(date '+%F %T')"
  /usr/bin/python3.10 -m metiswam4d_inspired_by_internw0.eval.iw0_campaign run "${args[@]}"
  status=$?
  echo "iw0 campaign manager exited with $status at $(date '+%F %T')"
  [ $status -eq 0 ] && break
  sleep 60
done
exit $status
