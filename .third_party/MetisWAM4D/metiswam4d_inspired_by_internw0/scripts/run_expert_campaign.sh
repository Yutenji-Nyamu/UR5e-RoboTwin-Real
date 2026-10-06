#!/usr/bin/env bash
# One host of a scripted-expert RoboDojo collection campaign.  Inside tmux:
#   tmux new-session -d -s expert_<name> "bash metiswam4d_inspired_by_internw0/scripts/run_expert_campaign.sh <out> <gpus> [clients]"
# Isaac clients run in the RoboDojo runtime (no policy server); the manager's watchdog keeps the GPU keep-alive up.
# A failed manager is restarted after 60 s (up to 20 times).
set -uo pipefail
OUT=$1
GPUS=$2
CLIENTS=${3:-}
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"

# shellcheck source=/dev/null
source /m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files/RoboDojo/scripts/activate_robodojo.sh || exit 2
cd "$PROJECT" || exit 2
export PYTHONPATH="$PROJECT" PYTHONDONTWRITEBYTECODE=1

args=(--output "$OUT" --gpus "$GPUS")
[ -n "$CLIENTS" ] && args+=(--clients "$CLIENTS")

status=1
for attempt in $(seq 1 20); do
  echo "expert campaign manager attempt $attempt at $(date '+%F %T')"
  /usr/bin/python3.10 -m metiswam4d_inspired_by_internw0.expert.expert_campaign run "${args[@]}"
  status=$?
  echo "expert campaign manager exited with $status at $(date '+%F %T')"
  [ $status -eq 0 ] && break
  sleep 60
done
exit $status
