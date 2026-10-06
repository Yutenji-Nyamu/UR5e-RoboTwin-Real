#!/usr/bin/env bash
# Separate Python 3.11 environment for the original InternW0-Delta runtime (teacher / reference evaluation).
# Pins follow XPolicyLab policy/InternW0_delta/install.sh + wam_runtime/pyproject.toml; the shared
# /usr/bin/python3.10 environment is not touched.  Run on a training machine inside tmux; idempotent.
set -euo pipefail
ENVS=/ytech_milm_intern/danglingwei/envs
VENV=$ENVS/internw0-delta
SRC=/m2v_intern_v3/danglingwei/files/InternW0_delta_src/XPolicyLab/policy/InternW0_delta
BASE_PY=$ENVS/uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
export UV_CACHE_DIR=$ENVS/.uv_cache UV_HTTP_TIMEOUT=300
export https_proxy=http://oversea-squid2.ko.txyun:11080 http_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH

mkdir -p "$ENVS"
[[ -x $VENV/bin/python ]] || uv venv "$VENV" --python "$BASE_PY"
PY=$VENV/bin/python
uv pip install --python "$PY" torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" \
  "modelscope==1.34.0" "ftfy==6.3.1" "flash-linear-attention==0.5.1" \
  "accelerate==1.14.0" "einops==0.8.1" "hydra-core==1.3.2" "numpy==1.26.4" "omegaconf==2.3.0" \
  "Pillow==11.3.0" "rich==14.2.0" "safetensors==0.8.0" "tqdm==4.66.5" "transformers==5.13.0" \
  "pyyaml>=6" "opencv-python-headless>=4.8" "h5py>=3.8" "websockets>=14.0" "msgpack>=1.0.8" \
  "msgpack-numpy>=0.4.8" "pydantic>=2.5" "huggingface_hub" "sentencepiece" "protobuf"
uv pip install --python "$PY" --no-build-isolation "causal-conv1d==1.6.2.post1" \
  || echo "WARN causal-conv1d not installed; runtime falls back with WAM_DISABLE_CAUSAL_CONV1D_FAST_PATH=1"
"$PY" -c "import torch, transformers, fla; print('OK', torch.__version__, torch.cuda.is_available(), transformers.__version__)"
