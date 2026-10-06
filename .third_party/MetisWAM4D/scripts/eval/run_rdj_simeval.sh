#!/usr/bin/env bash
# One host of the RoboDojo closed-loop campaign (metiswam4d/eval/rdj_campaign.py).  Run inside tmux on a machine with
# the RoboDojo Isaac runtime (/usr/local/robodojo-python-runtimes + the shared .robodojo_runtime site-packages):
#   tmux new-session -d -s rdjeval "bash scripts/eval/run_rdj_simeval.sh <out> <gpus> [clients per gpu] [keepalive pid file]"
# - The RoboDojo environment (Python 3.11 runtime, Isaac extension cache, EGL Vulkan ICD, oversea proxy for the
#   Isaac CDN assets) is activated for the Isaac clients; the policy servers run on /usr/bin/python3.10 with their own
#   PYTHONPATH (set by the campaign manager).
# - The GPU keep-alive (wangrunqi_nvml_busy.py) stays up: it is a CUDA sleep kernel that fills the utilisation gaps
#   the simulators leave; the manager's watchdog restarts it if it dies.
# - A manager that exits with an error is restarted after 60 s (round-level resume through the official manifest;
#   up to 20 times).
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
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com

args=(--output "$OUT" --gpus "$GPUS")
[ -n "$CLIENTS" ] && args+=(--clients "$CLIENTS")
[ -n "$KEEP_PID" ] && args+=(--keepalive-pid-file "$KEEP_PID")
[ -n "${RDJ_BASE_PORT:-}" ] && args+=(--base-port "$RDJ_BASE_PORT")

status=1
for attempt in $(seq 1 20); do
  echo "campaign manager attempt $attempt at $(date '+%F %T')"
  /usr/bin/python3.10 -m metiswam4d.eval.rdj_campaign run "${args[@]}"
  status=$?
  echo "campaign manager exited with $status at $(date '+%F %T')"
  [ $status -eq 0 ] && break
  sleep 60
done
exit $status
