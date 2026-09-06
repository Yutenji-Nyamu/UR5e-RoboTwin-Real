#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the RoboTwinSimReal Conda environment first." >&2
  exit 1
fi

python -m pip install --no-build-isolation --no-deps --editable "${PROJECT_ROOT}"

commands=(
  ur5e-real
  ur5e-collect-init
  ur5e-collect
  ur5e-replay-init
  ur5e-replay
  ur5e-policy-init
  ur5e-infer-init
  ur5e-infer
  ur5e-storage-repair
)

for command in "${commands[@]}"; do
  target="${CONDA_PREFIX}/bin/${command}"
  if [[ ! -x "${target}" ]]; then
    echo "Missing generated command: ${target}" >&2
    exit 1
  fi
done

sudo mkdir -p /usr/local/bin
for command in "${commands[@]}"; do
  sudo ln -sfn "${CONDA_PREFIX}/bin/${command}" "/usr/local/bin/${command}"
done

echo "Installed operator commands from ${CONDA_PREFIX}/bin into /usr/local/bin."
