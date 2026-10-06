#!/usr/bin/env bash
# Shared environment for MetisWAM4D training / verification scripts.
METIS_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METIS_PROJECT
export METIS_PYTHON="${METIS_PYTHON:-python}"
export PYTHONPATH="$METIS_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
# Vendored OpenWAM (only needed by scripts/verify_alpha_equivalence.py, never by training).
export OPENWAM_VENDOR="${OPENWAM_VENDOR:-/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824/janusact4d_rdj_imperfect_v1/vendor}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Network. BCC pods (t3/t4, dev box): eth0. IDC hosts (t5/t6): eth01 carries the routable 10.82.x address
# (~5.8 Gbps) while eth02-eth05 (26.1.x, routable between t5 and t6, no RDMA devices in the container)
# give ~25.7 Gbps for NCCL sockets (measured by the predecessor project). Override any variable to force.
if [[ -d /sys/class/net/eth02 && -d /sys/class/net/eth05 ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth02,eth03,eth04,eth05}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth01}"
  export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
  export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
else
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth01}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0,eth01}"
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
