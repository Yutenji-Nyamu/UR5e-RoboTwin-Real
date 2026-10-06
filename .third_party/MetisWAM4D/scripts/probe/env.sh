#!/usr/bin/env bash
# Environment for the probe scripts (Janus RT2 baseline + RoboTwin sim share /usr/bin/python 3.10).
export JANUS_P=/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824
export METIS_P=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921
export PYTHONPATH="$JANUS_P/janusact4d_rt2imperfect_v1/vendor:$JANUS_P:$METIS_P:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PROBE_OUT=/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes
export PY=/usr/bin/python
