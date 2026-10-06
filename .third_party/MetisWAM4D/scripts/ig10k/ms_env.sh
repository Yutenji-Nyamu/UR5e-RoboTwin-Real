# Environment for the Imitator-Game ManiSkill fork on /usr/bin/python3.10 (sapien 3.0.0b1, torch 2.7.1).
# Missing pure deps live in an overlay dir; the shared site-packages are not modified.
export IG_ROOT=/m2v_intern_v3/danglingwei/codes/TheImitatorGame_261005
export IG_OVERLAY=/ytech_milm_intern/danglingwei/envs/ig_py310_overlay
export PYTHONPATH=$IG_ROOT:$IG_OVERLAY${PYTHONPATH:+:$PYTHONPATH}
export MS_ASSET_DIR=${MS_ASSET_DIR:-$HOME/.maniskill}
# 10_nvidia.json (libEGL_nvidia) renders on dev and d1; nvidia_icd.json fails on d1 with ErrorIncompatibleDriver.
export VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
