#!/usr/bin/env bash
# GenManip (EBench server) + EBench bridge environment: Python 3.10 venv with Isaac Sim 4.1 pip wheels, torch 2.4
# (cu121), cuRobo, GenManip requirements, genmanip-client and the OpenWAM bridge dependencies. Mirrors
# GenManip/scripts/install.sh without conda. Kept apart from /usr/bin/python3.10 (training / policy env) and the
# RoboDojo Isaac 5.1 runtime.
#
# Lives under /usr/local/ebench on the node's local disk (Isaac wheels unpack tens of thousands of small files,
# which Ceph writes far too slowly). System-level state must persist through the dev machine's image, so the
# canonical copy is on the dev machine; training nodes get it from the image or via sync_ebench_env.sh.
# Requires the system library libturbojpeg (apt).
#   bash scripts/ebench/setup_genmanip_env.sh
set -euo pipefail

SUITE=/ytech_milm_intern/danglingwei/files/EBench_suite
LOCAL=/usr/local/ebench
VENV=$LOCAL/genmanip-venv
PROXY=http://oversea-squid1.jp.txyun:11080
export http_proxy=$PROXY https_proxy=$PROXY
export PIP_CACHE_DIR=$LOCAL/pip_cache
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p "$LOCAL"

[[ -x $VENV/bin/python ]] || /usr/bin/python3.10 -m venv "$VENV"
PY=$VENV/bin/python
$PY -m pip install -U pip setuptools wheel

$PY -m pip install isaacsim==4.1.0 isaacsim-extscache-physics==4.1.0 isaacsim-extscache-kit==4.1.0 \
    isaacsim-extscache-kit-sdk==4.1.0 --extra-index-url https://pypi.nvidia.com
$PY -m pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121

# GenManip uses the cuRobo 0.7 API (curobo.types.state, ...); main is a rewritten major version.
if ! $PY -c "import curobo.types.state" 2>/dev/null; then
  $PY -m pip uninstall -y nvidia-curobo 2>/dev/null || true
  rm -rf "$LOCAL/curobo"
  git -c http.proxy=$PROXY -c https.proxy=$PROXY clone --depth 1 --branch v0.7.8 \
    https://github.com/NVlabs/curobo.git "$LOCAL/curobo"
  TORCH_CUDA_ARCH_LIST=8.0 $PY -m pip install -e "$LOCAL/curobo" --no-build-isolation
fi

$PY -m pip install -r "$SUITE/GenManip/requirements.txt" fastapi
$PY -m pip install -e "$SUITE/genmanip-client"
# PyTurboJPEG 2.x needs libjpeg-turbo 3.x; Ubuntu 22.04 ships 2.1.
# Isaac Sim 4.1 / mplib need numpy<2; opencv >= 4.11 would pull numpy 2.
$PY -m pip install "PyTurboJPEG<2" "websockets>=15" "opencv-python-headless<4.11" "numpy<2" filelock
# Isaac's prebundled botocore 1.34.68 shadows the venv's at runtime; boto3 / s3transfer must match it or
# omni.replicator.core (camera annotators) fails to load.
$PY -m pip install boto3==1.34.68 botocore==1.34.68 "s3transfer<0.11"

$PY - <<'EOF'
import torch, curobo, genmanip_client, turbojpeg, websockets, cv2
turbojpeg.TurboJPEG()
print("torch", torch.__version__, torch.cuda.is_available(), "| curobo, genmanip_client, turbojpeg, websockets ok")
EOF
$PY -c "import isaacsim; print('isaacsim ok')"
