#!/usr/bin/env bash
# Unattended tail for the IG-10K annotation set. Run on t6 (GPU host):
#   wait until all 69 task dirs have masks -> repair any depth mkv deleted by the old OOM path
#   -> final track4d sweep on 6 GPUs -> manifest.parquet.  Logs to _logs/finish.log.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A=/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/human
PY=/usr/bin/python3.10
LOG=$A/_logs/finish.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd "$PROJ"

log "waiting for 69/69 masks + depth"
while true; do
  m=$(ls $A/*/*/masks.h5 2>/dev/null | wc -l); d=$(ls $A/*/*/depth_index.json 2>/dev/null | wc -l)
  [ "$m" -ge 69 ] && [ "$d" -ge 69 ] && break
  sleep 300
done
log "masks+depth complete; stopping mask watchdogs"
bash scripts/data_prep/watchdog_ig10k.sh stop >/dev/null 2>&1 || true
ssh -o ConnectTimeout=10 t5 'bash /m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/data_prep/watchdog_ig10k.sh stop' >/dev/null 2>&1 || true

log "repair-depth (old OOM path may have deleted finished depth on the last dirs)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/data_prep/ig10k_human_depth_masks.py repair-depth >> "$LOG" 2>&1

log "final track4d sweep on 6 GPUs"
rm -rf $A/_locks_track4d
for g in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup $PY scripts/data_prep/ig10k_human_track4d.py run --shuffle \
    >> $A/_logs/track4d_finish_gpu$g.out 2>&1 < /dev/null &
  sleep 1
done
while [ "$(ps -eo args | grep -c '[i]g10k_human_track4d.py')" -gt 0 ]; do sleep 120; done
log "track4d dirs: $(ls $A/*/*/track4d.h5 | wc -l)/69"

log "manifest"
/usr/bin/python3 scripts/data_prep/ig10k_anno_reader.py manifest >> "$LOG" 2>&1
log "FINISHED"
