#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
if ! command -v "${UV_BIN}" >/dev/null && [[ -x "${PROJECT_ROOT}/.venv/bootstrap/bin/uv" ]]; then
  UV_BIN="${PROJECT_ROOT}/.venv/bootstrap/bin/uv"
fi
PI05_ENV="${PROJECT_ROOT}/.venv/pi05"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.venv/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${PROJECT_ROOT}/.venv/python"
export GIT_LFS_SKIP_SMUDGE=1
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
if ! command -v "${UV_BIN}" >/dev/null; then
  echo "Install uv or set UV_BIN to its executable; the DP environment is not modified." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/.third_party/RoboTwin/policy/pi05/src/openpi/models/pi0.py" ]]; then
  echo "Bootstrap the locked RoboTwin checkout first: scripts/bootstrap_robotwin.sh" >&2
  exit 1
fi
if [[ ! -x "${PI05_ENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.11 "${PI05_ENV}"
fi
requirements="${PROJECT_ROOT}/integrations/pi05/requirements.lock"
if [[ ! -f "${requirements}" ]]; then
  requirements="${PROJECT_ROOT}/integrations/pi05/requirements.in"
fi
"${UV_BIN}" --quiet pip sync --python "${PI05_ENV}/bin/python" "${requirements}"
"${UV_BIN}" pip install --python "${PI05_ENV}/bin/python" --no-deps --editable "${PROJECT_ROOT}"
"${UV_BIN}" pip check --python "${PI05_ENV}/bin/python"
echo "pi05 model environment: ${PI05_ENV}/bin/python"
