#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The patched upstream source now ships in this repository, not a nested clone.
# Run directly from src so this check needs only Python's standard library.
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN:-python3}" -m ur5e_real.vendor RoboTwin
