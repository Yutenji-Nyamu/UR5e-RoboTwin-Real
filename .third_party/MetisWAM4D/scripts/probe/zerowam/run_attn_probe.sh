#!/usr/bin/env bash
# Zero-WAM attention probe: server with attention capture + a few client episodes.  Usage: run_attn_probe.sh <gpu> <condition> <test_num> <task...>
set -uo pipefail
GPU=$1; COND=$2; TEST_NUM=$3; shift 3
ZW=/m2v_intern_v3/danglingwei/codes/Zero-WAM_260923
ROBOTWIN_ROOT=$ZW/third_party/RoboTwin
HERE=/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/probe/zerowam
PROBE_OUT=/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/probes/zerowam
LATENTS=/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/HumanGen/human_latents/robotwin
export MODEL_PATH=/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/ZeroWAM/zero-wam-posttrain-robotwin
export PATH=/root/py310bin:$PATH PYTHONPATH=$ZW:$ROBOTWIN_ROOT
export ZW_PROBE_OUT=$PROBE_OUT/attn/$COND
PORT=$((29376 + GPU)); MASTER_PORT=$((29481 + GPU))
case $COND in official) MAP=$ZW/evaluation/robotwin/robotwin_icl_human_videos.py; NOICL=0;; swap_task) MAP=$HERE/maps/swap_task.py; NOICL=0;; no_icl) MAP=$ZW/evaluation/robotwin/robotwin_icl_human_videos.py; NOICL=1;; esac
mkdir -p $PROBE_OUT/logs $ZW_PROBE_OUT
LOG=$PROBE_OUT/logs/attnserver_${COND}_gpu${GPU}.log
cd $ZW
CUDA_VISIBLE_DEVICES=$GPU ICL_CFG=5 TARGET_TEXT_CFG=-1 nohup python -m torch.distributed.run --nproc_per_node 1 --master_port $MASTER_PORT $HERE/zerowam_attn_server.py --config-name robotwin --port $PORT --save_root $PROBE_OUT/visualization/attn_$COND > $LOG 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do grep -q "server listening" $LOG 2>/dev/null && break; sleep 5; done
grep -q "server listening" $LOG || { echo "server failed"; tail -30 $LOG; kill $SERVER_PID; exit 1; }
echo "[$(date)] probe server ready on $PORT"
cd $ROBOTWIN_ROOT
for TASK in "$@"; do
  echo "[$(date)] === attn-probe $COND / $TASK (n=$TEST_NUM)"
  ZEROWAM_NO_ICL=$NOICL LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-} VK_ICD_FILENAMES=/usr/local/lib/python3.10/dist-packages/sapien/vulkan_library/10_nvidia.json CUDA_VISIBLE_DEVICES=$GPU PYTHONWARNINGS=ignore::UserWarning \
  python $HERE/zerowam_client.py --config $ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml --port $PORT --save_root $PROBE_OUT/attn_eval/$COND \
    --video_guidance_scale -1 --action_guidance_scale 1 --icl_guidance_scale 5 --icl_human_video_map $MAP --icl_latent_root $LATENTS --icl_seed 0 --test_num $TEST_NUM \
    --overrides --task_name $TASK --task_config demo_clean --train_config_name 0 --model_name 0 --ckpt_setting 0 --seed 0 --policy_name ACT > $PROBE_OUT/logs/attnclient_${COND}_${TASK}.log 2>&1
  grep -h "Success rate" $PROBE_OUT/logs/attnclient_${COND}_${TASK}.log | tail -1
done
kill $SERVER_PID 2>/dev/null; sleep 5; kill -9 $SERVER_PID 2>/dev/null
echo "[$(date)] done"
