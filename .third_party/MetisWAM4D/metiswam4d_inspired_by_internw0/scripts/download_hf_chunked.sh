#!/usr/bin/env bash
# Chunked, parallel, resumable download of one Hugging Face file with sha256 verification.
#
#   bash download_hf_chunked.sh <repo> <file> <size> <sha256> <dest_dir>
#
# huggingface.co is reached through the overseas proxy (2026-10-04 on t2: 1 stream ~1 MB/s, 16 streams ~17 MB/s).
# Every part is fetched with HTTP Range and appended until it reaches its length, so re-running resumes.
# Env knobs: NPARTS (96), PAR (48), PROXY.
set -uo pipefail

REPO=$1; FILE=$2; SIZE=$3; SHA=$4; DEST=$5
NPARTS="${NPARTS:-96}"
PAR="${PAR:-48}"
export https_proxy="${PROXY:-http://oversea-squid2.ko.txyun:11080}" http_proxy="${PROXY:-http://oversea-squid2.ko.txyun:11080}"
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

URL="https://huggingface.co/$REPO/resolve/main/$FILE"
OUT="$DEST/$FILE"
PARTDIR="$DEST/.parts_$(basename "$FILE")"
log() { echo "[$(date '+%F %T')] $*"; }
parts_bytes() { stat -c %s "$PARTDIR"/part-??? "$PARTDIR"/part-???.tmp 2>/dev/null | awk '{s+=$1} END{printf "%d\n", s}'; }

mkdir -p "$(dirname "$OUT")" "$PARTDIR"
log "START $REPO/$FILE size=$SIZE -> $OUT nparts=$NPARTS par=$PAR"

if [[ -f "$OUT" && "$(stat -c %s "$OUT")" == "$SIZE" ]]; then
  log "already present with expected size"
else
  CHUNK=$(( (SIZE + NPARTS - 1) / NPARTS ))
  fetch_part() {
    local i=$1 a b len p f have code tries=0
    a=$(( i * CHUNK )); b=$(( a + CHUNK - 1 )); (( b >= SIZE )) && b=$(( SIZE - 1 )); len=$(( b - a + 1 ))
    (( len <= 0 )) && return 0
    p=$(printf '%s/part-%03d' "$PARTDIR" "$i"); f="$p.tmp"
    while :; do
      have=$(stat -c %s "$p" 2>/dev/null || echo 0)
      (( have == len )) && return 0
      if (( have > len )); then rm -f "$p"; have=0; fi
      (( tries++ )); if (( tries > 300 )); then log "part $i GAVE UP"; return 1; fi
      rm -f "$f"
      code=$(curl -sL --connect-timeout 30 --speed-time 60 --speed-limit 20000 \
                  -r $(( a + have ))-$b -o "$f" -w '%{http_code}' "$URL")
      if [[ "$code" == "206" && -s "$f" ]]; then cat "$f" >> "$p"; else log "part $i http=$code retry"; sleep 10; fi
      rm -f "$f"
    done
  }
  export -f fetch_part log; export URL PARTDIR CHUNK SIZE
  ( prev=$(parts_bytes); while sleep 60; do cur=$(parts_bytes); log "progress $((cur/1048576))/$((SIZE/1048576)) MiB  $(( (cur-prev)/60/1048576 )) MiB/s"; prev=$cur; done ) &
  PROG=$!
  seq 0 $((NPARTS-1)) | xargs -P "$PAR" -I{} bash -c 'fetch_part {}'
  kill "$PROG" 2>/dev/null
  for i in $(seq 0 $((NPARTS-1))); do
    a=$(( i * CHUNK )); b=$(( a + CHUNK - 1 )); (( b >= SIZE )) && b=$(( SIZE - 1 )); (( b < a )) && continue
    p=$(printf '%s/part-%03d' "$PARTDIR" "$i")
    [[ "$(stat -c %s "$p" 2>/dev/null || echo 0)" == "$(( b - a + 1 ))" ]] || { log "FAILED part $i incomplete; re-run to resume"; exit 1; }
  done
  ls "$PARTDIR"/part-??? | sort | xargs cat > "$OUT.tmp" && mv -f "$OUT.tmp" "$OUT"
  [[ "$(stat -c %s "$OUT")" == "$SIZE" ]] || { log "FAILED size mismatch"; exit 1; }
  rm -rf "$PARTDIR"
fi

got=$(sha256sum "$OUT" | awk '{print $1}')
if [[ "$got" == "$SHA" ]]; then echo "$got  $(basename "$FILE")" > "$OUT.sha256"; log "SHA256 OK $OUT"
else log "SHA256 MISMATCH got=$got expected=$SHA"; exit 1; fi
