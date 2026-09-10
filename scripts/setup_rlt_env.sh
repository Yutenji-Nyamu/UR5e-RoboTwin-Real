#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLT_UV="${UV_BIN:-${PROJECT_ROOT}/.venv/bootstrap/bin/uv}"
RLT_ENV="${PROJECT_ROOT}/.venv/rlt"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.venv/uv-cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
if [[ ! -x "${RLT_UV}" ]]; then
  echo "Install uv or set UV_BIN; bootstrap scripts/setup_pi05.sh first." >&2
  exit 1
fi
if [[ ! -x "${PROJECT_ROOT}/.venv/pi05/bin/python" ]]; then
  echo "Set up the successful native pi05 environment first." >&2
  exit 1
fi
if [[ ! -x "${RLT_ENV}/bin/python" ]]; then
  "${RLT_UV}" venv --python "${PROJECT_ROOT}/.venv/pi05/bin/python" "${RLT_ENV}"
fi
"${RLT_UV}" pip sync --python "${RLT_ENV}/bin/python" "${PROJECT_ROOT}/integrations/rlt/requirements.lock"
"${RLT_UV}" pip install --python "${RLT_ENV}/bin/python" --no-deps --editable "${PROJECT_ROOT}"
"${RLT_UV}" pip check --python "${RLT_ENV}/bin/python"
echo "RLT learner/service: ${RLT_ENV}/bin/python; hardware stays in RoboTwinSimReal."
