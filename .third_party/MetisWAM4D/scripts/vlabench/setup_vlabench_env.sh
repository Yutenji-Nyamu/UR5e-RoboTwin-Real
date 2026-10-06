#!/usr/bin/env bash
# VLABench (MuJoCo) env: separate venv because VLABench pins numpy 1.25 / mujoco 3.2.2 / dm_control 1.0.22.
# Install on the dev machine (image-persisted /usr/local), then sync to training nodes with sync_vlabench_env.sh.
set -euo pipefail

VENV=${VENV:-/usr/local/vlabench/venv}
VLABENCH_ROOT_DIR=${VLABENCH_ROOT_DIR:-/ytech_milm_intern/danglingwei/files/VLABench}
PROXY=${PROXY:-http://oversea-squid1.jp.txyun:11080}
export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY
export UV_CACHE_DIR=${UV_CACHE_DIR:-/usr/local/vlabench/uv_cache}

mkdir -p "$(dirname "$VENV")"
[ -x "$VENV/bin/python" ] || uv venv --python /usr/bin/python3.10 "$VENV"
PY="$VENV/bin/python"

uv pip install --python "$PY" \
  'numpy==1.25.0' 'scipy==1.14.0' 'mujoco==3.2.2' 'dm_control==1.0.22' \
  'opencv-python-headless<4.12' 'gym==0.26.2' 'gymnasium==0.29.1' \
  'mediapy==1.2.0' 'h5py==3.11.0' 'imageio>=2.34' 'imageio-ffmpeg' 'av<14' \
  'Pillow>=10' 'PyYAML>=6' 'websockets>=16,<17' 'open3d==0.18.0' \
  colorlog colorama openai 'rtree==1.2.0' networkx gdown tqdm pandas pyarrow \
  scikit-learn trimesh

uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'

[ -d "$VLABENCH_ROOT_DIR/src/rrt-algorithms" ] || git -c http.proxy="$PROXY" clone --depth 1 \
  https://github.com/motion-planning/rrt-algorithms.git "$VLABENCH_ROOT_DIR/src/rrt-algorithms"
uv pip install --python "$PY" --no-deps --no-build-isolation -e "$VLABENCH_ROOT_DIR/src/rrt-algorithms"
uv pip install --python "$PY" --no-deps -e "$VLABENCH_ROOT_DIR"

MUJOCO_GL=egl "$PY" -c "import numpy, mujoco, dm_control, torch, VLABench; print('ok', numpy.__version__, mujoco.__version__, torch.__version__)"
