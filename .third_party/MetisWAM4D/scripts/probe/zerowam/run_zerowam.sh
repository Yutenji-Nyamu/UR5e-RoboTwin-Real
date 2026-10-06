#!/usr/bin/env bash
# Zero-WAM demo-sensitivity probe: one GPU = one server + sequential clients.
# Usage: run_zerowam.sh <gpu> <condition> <test_num> <task1> [task2 ...]
#   condition: official | swap_task | seen_task | no_icl
# Results: $PROBE_OUT/zerowam/<condition>/<task>/  (RoboTwin eval_result layout written by the client)
set -uo pipefail
GPU=$1; COND=$2; TEST_NUM=$3; shift 3
ZW=/m2v_intern_v3/danglingwei/codes/Zero-WAM_260923
ROBOTWIN_ROOT=$ZW/third_party/RoboTwin
MAPS=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/probe/zerowam/maps
PROBE_OUT=/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes/zerowam
LATENTS=/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/HumanGen/human_latents/robotwin
export MODEL_PATH=/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/ZeroWAM/zero-wam-posttrain-robotwin
export PATH=/root/py310bin:$PATH
PORT=$((29056 + GPU)); MASTER_PORT=$((29161 + GPU))
mkdir -p $PROBE_OUT/logs $PROBE_OUT/$COND
LOG=$PROBE_OUT/logs/server_${COND}_gpu${GPU}.log

case $COND in
  official)  MAP=$ZW/evaluation/robotwin/robotwin_icl_human_videos.py; NOICL=0 ;;
  swap_task) MAP=$MAPS/swap_task.py; NOICL=0 ;;
  seen_task) MAP=$MAPS/seen_task.py; NOICL=0 ;;
  no_icl)    MAP=$ZW/evaluation/robotwin/robotwin_icl_human_videos.py; NOICL=1 ;;
  *) echo "unknown condition $COND"; exit 1 ;;
esac

cd $ZW
CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT MASTER_PORT=$MASTER_PORT SAVE_ROOT=$PROBE_OUT/visualization/$COND ICL_CFG=5 TARGET_TEXT_CFG=-1 \
  nohup bash evaluation/robotwin/launch_server.sh > $LOG 2>&1 &
SERVER_PID=$!
echo "[$(date)] server pid $SERVER_PID port $PORT log $LOG"
for i in $(seq 1 120); do grep -q "server listening" $LOG 2>/dev/null && break; sleep 5; done
grep -q "server listening" $LOG || { echo "server failed to start"; tail -20 $LOG; kill $SERVER_PID; exit 1; }
echo "[$(date)] server ready"

cd $ROBOTWIN_ROOT
for TASK in "$@"; do
  echo "[$(date)] === $COND / $TASK (n=$TEST_NUM)"
  ZEROWAM_NO_ICL=$NOICL PYTHONPATH=$ZW:$ROBOTWIN_ROOT LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-} \
  VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json CUDA_VISIBLE_DEVICES=$GPU PYTHONWARNINGS=ignore::UserWarning \
  python /m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/probe/zerowam/zerowam_client.py \
    --config $ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml --port $PORT --save_root $PROBE_OUT/$COND \
    --video_guidance_scale -1 --action_guidance_scale 1 --icl_guidance_scale 5 \
    --icl_human_video_map $MAP --icl_latent_root $LATENTS --icl_seed 0 --test_num $TEST_NUM \
    --overrides --task_name $TASK --task_config demo_clean --train_config_name 0 --model_name 0 --ckpt_setting 0 --seed 0 --policy_name ACT \
    > $PROBE_OUT/logs/client_${COND}_${TASK}.log 2>&1
  echo "[$(date)] done $TASK rc=$?"; grep -h "success rate\|Success rate\|suc_rate\|SUCCESS" $PROBE_OUT/logs/client_${COND}_${TASK}.log | tail -2
done
kill $SERVER_PID 2>/dev/null
echo "[$(date)] server stopped"
