#!/usr/bin/env bash
# Copy /usr/local/ebench (GenManip venv + built cuRobo) between machines with the same base image, keeping the
# path so the venv stays valid. Run on the receiving machine:
#   bash scripts/ebench/sync_ebench_env.sh t1      # pull from t1 into this machine
set -euo pipefail
SRC=${1:?source host alias}
rm -rf /usr/local/ebench
ssh "$SRC" "tar -C /usr/local --exclude=ebench/pip_cache -cf - ebench" | tar -C /usr/local -xf -
OMNI_KIT_ACCEPT_EULA=YES /usr/local/ebench/genmanip-venv/bin/python -c \
  "import torch, curobo, genmanip_client, turbojpeg, isaacsim; turbojpeg.TurboJPEG(); print('ebench env ok', torch.__version__)"
