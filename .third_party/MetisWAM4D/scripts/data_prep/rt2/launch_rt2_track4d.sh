#!/usr/bin/env bash
# Launch a range of RT2 Track4D shards on THIS node (8 GPUs), plus one GPU filler per GPU so device
# utilisation stays high while the (bursty) SAPIEN replays run.
#   bash scripts/data_prep/rt2/launch_rt2_track4d.sh VARIANTS NUM_SHARDS FIRST_SHARD LAST_SHARD
#   e.g. t4: ... all 64 0 31      t3: ... all 64 32 63
# VARIANTS: demo_clean_4d | demo_randomized_4d | all (clean first, then randomized). Resumable.
set -euo pipefail
variant="${1:?variants}"; num_shards="${2:-64}"; first="${3:-0}"; last="${4:-$((num_shards - 1))}"
filler="${FILLER:-1}"   # FILLER=0 when the node's GPUs are already busy with other work
PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT=/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D
mkdir -p "$OUT/_logs"
if [[ "$variant" == "all" ]]; then variants=(demo_clean_4d demo_randomized_4d); else variants=("$variant"); fi
gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
host="$(hostname -s)"
# One shared pending list per launch (all nodes of the same launch must use the same file: run this
# on the first node, or set PENDING=0 on the others once it exists).
if [[ "${PENDING:-1}" == "1" ]]; then
  /usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py pending --variants "${variants[@]}"
fi
# IDC hosts (t5/t6) have no system-wide NVIDIA Vulkan ICD; point SAPIEN at its bundled one.
vk=""
if [[ ! -e /etc/vulkan/icd.d/nvidia_icd.json && ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  vk="VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:\${LD_LIBRARY_PATH:-}"
fi
extra=""; suffix=""
if [[ "${REVERSE:-0}" == "1" ]]; then extra="--reverse"; suffix="r"; fi   # mirror node: consume blocks from the back
for ((shard = first; shard <= last; shard++)); do
  gpu=$(( shard % gpus ))
  name="rt2track_s${shard}${suffix}"
  tmux kill-session -t "$name" 2>/dev/null || true
  # Stagger start-up so dozens of processes do not compile the planner's CUDA kernels at once.
  tmux new-session -d -s "$name" \
    "sleep $(( (shard - first) * 4 )); cd $PROJECT && env $vk CUDA_VISIBLE_DEVICES=$gpu PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 nice -n 5 /usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py run --shard $shard --num-shards $num_shards --device cuda:0 --variants ${variants[*]} $extra > $OUT/_logs/${host}_${name}.out 2>&1"
done
if [[ "$filler" == "1" ]] && ! tmux has-session -t gpu_filler 2>/dev/null; then
  tmux new-session -d -s gpu_filler \
    "cd $PROJECT && /usr/bin/python3.10 scripts/data_prep/gpu_filler.py --duty 0.92 > $OUT/_logs/${host}_gpu_filler.out 2>&1"
fi
echo "$(tmux ls | grep -c rt2track_) shards + filler on $host"
echo "status: /usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py status --variants ${variants[*]}"
