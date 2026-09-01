#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/robotwin.lock"

VENDOR_ROOT="${PROJECT_ROOT}/.third_party"
ROBOTWIN_ROOT="${VENDOR_ROOT}/RoboTwin"
PATCH_FILE="${PROJECT_ROOT}/integrations/robotwin/patches/0001-act-normalize-num-queries.patch"

mkdir -p "${VENDOR_ROOT}"
if [[ ! -d "${ROBOTWIN_ROOT}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${ROBOTWIN_REPOSITORY}" "${ROBOTWIN_ROOT}"
fi

if [[ -n "$(git -C "${ROBOTWIN_ROOT}" status --porcelain)" ]]; then
  if ! git -C "${ROBOTWIN_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Refusing to replace a modified RoboTwin checkout: ${ROBOTWIN_ROOT}" >&2
    exit 1
  fi
fi

git -C "${ROBOTWIN_ROOT}" fetch origin "${ROBOTWIN_COMMIT}"
git -C "${ROBOTWIN_ROOT}" checkout --detach "${ROBOTWIN_COMMIT}"

if git -C "${ROBOTWIN_ROOT}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${ROBOTWIN_ROOT}" apply "${PATCH_FILE}"
fi

actual_commit="$(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${ROBOTWIN_COMMIT}" ]]; then
  echo "RoboTwin commit mismatch: expected ${ROBOTWIN_COMMIT}, got ${actual_commit}" >&2
  exit 1
fi

echo "RoboTwin ready: ${ROBOTWIN_ROOT} @ ${actual_commit}"
echo "Local adapter patch applied; upstream checkout remains ignored by this repository."
