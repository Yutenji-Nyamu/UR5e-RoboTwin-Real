#!/usr/bin/env bash
# Download OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model (checkpoint_step_154000.safetensors, 24.8 GB,
# + config.yaml / README.md / tokenizer/) into the local model zoo.
#
# Method (measured 2026-09-21 on t5): hf-mirror.com -> cas-bridge.xethub.hf.co, DIRECT (no proxy).
#   1 stream ~3 MB/s, 8 -> 30 MB/s, 16 -> 60 MB/s, 32 -> ~100 MB/s, all 206, no 429.
#   huggingface.co is unreachable from t5/t6 (direct and via proxy 10.66.29.113:11080); hf-mirror via that
#   proxy is slower (12.5 MB/s @8 streams).  t6 cannot reach cas-bridge directly at all (only via proxy,
#   7.6 MB/s @8) -> run this on t5.  aria2c / hf_transfer are not installed on t5/t6, so the big file is
#   fetched as NPARTS byte-range chunks with curl (PAR in parallel; each chunk resumable), concatenated,
#   and sha256-verified against the HF LFS oid (x-linked-etag).  Small files via huggingface_hub, Xet off
#   (Xet CAS through the mirror returns 401, see docs/2026-09-18-IG-10K数据集调研说明.md §9).
#
# Idempotent / resumable: re-run to resume.  Finished parts are skipped; if the final file already exists
# with the right size it is only (re)verified.
#
# Launch (on t5):
#   LOG=/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/_download_foundation.log
#   tmux new-session -d -s hf_download "bash /m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921/scripts/download_alpha_foundation.sh >> $LOG 2>&1"
#   monitor: tail -f $LOG   |   tmux attach -t hf_download   |   stop: tmux kill-session -t hf_download
# Env knobs: DEST, NPARTS (64), PAR (32), VERIFY=0 to skip the final sha256 pass.
set -uo pipefail

REPO=OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model
BIG=checkpoint_step_154000.safetensors
BIG_SIZE=24813767464
BIG_SHA256=180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d   # x-linked-etag, repo commit 52df4e66
DEST="${DEST:-/m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model}"
NPARTS="${NPARTS:-64}"
PAR="${PAR:-32}"
VERIFY="${VERIFY:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

URL="$HF_ENDPOINT/$REPO/resolve/main/$BIG"
PARTDIR="$DEST/.parts_$BIG"
log() { echo "[$(date '+%F %T')] $*"; }
# bytes on disk for the big file: finished parts + in-flight .tmp bodies (printf %d avoids awk's 1e+09 notation)
parts_bytes() { stat -c %s "$PARTDIR"/part-??? "$PARTDIR"/part-???.tmp 2>/dev/null | awk '{s+=$1} END{printf "%d\n", s}'; }

mkdir -p "$DEST" "$PARTDIR"
log "START host=$(hostname -s) dest=$DEST endpoint=$HF_ENDPOINT nparts=$NPARTS par=$PAR"

# ---------- 1. small files (config.yaml, README.md, tokenizer/) ----------
log "fetching small files via huggingface_hub.snapshot_download"
python - "$REPO" "$DEST" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:3]
snapshot_download(repo, local_dir=dest, max_workers=4,
                  allow_patterns=["config.yaml", "README.md", ".gitattributes", "tokenizer/*", "*.npy", "*.json"])
EOF
log "small files rc=$?"

# ---------- 2. big file, chunked + parallel + resumable ----------
if [[ -f "$DEST/$BIG" && "$(stat -c %s "$DEST/$BIG")" == "$BIG_SIZE" ]]; then
  log "$BIG already present with expected size ($BIG_SIZE)"
else
  rs=$(curl -sIL --max-time 60 "$URL" | grep -i '^x-linked-size:' | tr -dc '0-9')
  [[ "$rs" == "$BIG_SIZE" ]] || log "WARNING remote x-linked-size='$rs' != $BIG_SIZE (continuing with $BIG_SIZE)"
  CHUNK=$(( (BIG_SIZE + NPARTS - 1) / NPARTS ))

  fetch_part() {   # $1 = part index; appends verified 206 bodies to part-NNN until it reaches its length
    local i=$1 a b len p f have code tries=0
    a=$(( i * CHUNK )); b=$(( a + CHUNK - 1 )); (( b >= BIG_SIZE )) && b=$(( BIG_SIZE - 1 )); len=$(( b - a + 1 ))
    p=$(printf '%s/part-%03d' "$PARTDIR" "$i"); f="$p.tmp"
    while :; do
      have=$(stat -c %s "$p" 2>/dev/null || echo 0)
      (( have == len )) && return 0
      if (( have > len )); then log "part $i oversize ($have>$len), restarting it"; rm -f "$p"; have=0; fi
      (( tries++ )); if (( tries > 300 )); then log "part $i GAVE UP after $tries tries"; return 1; fi
      rm -f "$f"
      code=$(curl -sL --connect-timeout 30 --speed-time 60 --speed-limit 20000 \
                  -r $(( a + have ))-$b -o "$f" -w '%{http_code}' "$URL")
      if [[ "$code" == "206" && -s "$f" ]]; then
        cat "$f" >> "$p"
      else
        log "part $i http=$code bytes=$(stat -c %s "$f" 2>/dev/null || echo 0) -> retry in 15s (try $tries)"
        sleep 15
      fi
      rm -f "$f"
    done
  }
  export -f fetch_part log; export URL PARTDIR CHUNK BIG_SIZE

  log "downloading $BIG ($BIG_SIZE bytes) as $NPARTS x ~$((CHUNK/1048576)) MiB parts, $PAR in parallel; have $(( $(parts_bytes)/1048576 )) MiB"
  ( prev=$(parts_bytes); while sleep 30; do cur=$(parts_bytes); log "progress $((cur/1048576))/$((BIG_SIZE/1048576)) MiB ($((cur*100/BIG_SIZE))%)  $(( (cur-prev)/30/1048576 )) MiB/s  ETA $(( cur>prev ? (BIG_SIZE-cur)*30/(cur-prev)/60 : -1 )) min"; prev=$cur; done ) &
  PROG=$!
  seq 0 $((NPARTS-1)) | xargs -P "$PAR" -I{} bash -c 'fetch_part {}'
  kill "$PROG" 2>/dev/null

  # verify every part length, then concatenate
  ok=1
  for i in $(seq 0 $((NPARTS-1))); do
    a=$(( i * CHUNK )); b=$(( a + CHUNK - 1 )); (( b >= BIG_SIZE )) && b=$(( BIG_SIZE - 1 ))
    p=$(printf '%s/part-%03d' "$PARTDIR" "$i")
    [[ "$(stat -c %s "$p" 2>/dev/null || echo 0)" == "$(( b - a + 1 ))" ]] || { log "part $i incomplete"; ok=0; }
  done
  if (( ! ok )); then log "FAILED: parts incomplete, re-run this script to resume"; exit 1; fi
  log "all $NPARTS parts complete, concatenating -> $DEST/$BIG"
  ls "$PARTDIR"/part-??? | sort | xargs cat > "$DEST/$BIG.tmp" && mv -f "$DEST/$BIG.tmp" "$DEST/$BIG"
  got=$(stat -c %s "$DEST/$BIG")
  if [[ "$got" != "$BIG_SIZE" ]]; then log "FAILED: concatenated size $got != $BIG_SIZE"; exit 1; fi
  rm -rf "$PARTDIR"
  log "concat done, size OK ($got)"
fi

# ---------- 3. integrity ----------
if [[ "$VERIFY" == "1" ]]; then
  log "sha256sum $BIG (expect $BIG_SHA256) ..."
  got=$(sha256sum "$DEST/$BIG" | awk '{print $1}')
  if [[ "$got" == "$BIG_SHA256" ]]; then
    echo "$got  $BIG" > "$DEST/$BIG.sha256"; log "SHA256 OK -> $DEST/$BIG.sha256"
  else
    log "SHA256 MISMATCH got=$got expected=$BIG_SHA256 -- delete $DEST/$BIG and re-run"; exit 1
  fi
fi
log "DONE"; ls -la "$DEST" "$DEST/tokenizer/google/umt5-xxl" 2>/dev/null
