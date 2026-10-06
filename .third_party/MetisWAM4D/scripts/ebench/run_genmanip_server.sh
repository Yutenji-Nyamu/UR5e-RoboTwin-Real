#!/usr/bin/env bash
# Local GenManip (EBench) evaluation server. Isaac workers are created on demand, one per client worker id, and
# Ray places them over all GPUs of the node (0.25 GPU each). Every GPU stays visible to every worker: Ray assigns
# a GPU id without setting CUDA_VISIBLE_DEVICES, and the (patched) worker passes it to Isaac as active/physics GPU.
#   PORT=8087 bash scripts/ebench/run_genmanip_server.sh [extra ray_eval_server.py args]
# The nodes run a shared NVIDIA MPS daemon, and every CUDA process on a node must be on the same side of it (see
# launch_alpha_eval.sh). EBENCH_USE_MPS=1 keeps the Isaac workers in MPS; otherwise an empty pipe directory makes
# them plain CUDA clients.
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
if [[ ${EBENCH_USE_MPS:-0} != 1 ]]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/ebench-no-mps
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
fi

GENMANIP=/ytech_milm_intern/danglingwei/files/EBench_suite/GenManip
PY=/usr/local/ebench/genmanip-venv/bin/python
PORT=${PORT:-8087}
# A800 has no RT cores: the default real-time RTX mode renders with heavy grain (the training data is clean).
# Path tracing with one sample per frame, no accumulation, and the OptiX denoiser is clean at ~48 ms per camera.
RENDER_SPP=${RENDER_SPP:-1}
export GENMANIP_CARB_SETTINGS=${GENMANIP_CARB_SETTINGS:-"{\"/rtx/rendermode\": \"PathTracing\", \"/rtx/pathtracing/spp\": $RENDER_SPP, \"/rtx/pathtracing/totalSpp\": $RENDER_SPP, \"/rtx/pathtracing/optixDenoiser/enabled\": true}"}

source "$(dirname "$0")/isaac_env.sh"
export OMP_NUM_THREADS=4
unset http_proxy https_proxy

cd "$GENMANIP"
exec "$PY" ray_eval_server.py --host 0.0.0.0 --port "$PORT" --no_save_process "$@"
