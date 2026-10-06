#!/usr/bin/env bash
# Parallel byte-range download of one large HF file through a proxy, resumable per part, sha256-verified.
#   bash fetch_sharded.sh <url> <dest_file> <size> <sha256>
# Env: PROXY (oversea squid), NPARTS (48), PAR (16).
set -uo pipefail
URL=$1; DEST=$2; SIZE=$3; SHA=$4
PROXY=${PROXY:-http://oversea-squid1.jp.txyun:11080}
NPARTS=${NPARTS:-48}; PAR=${PAR:-16}
PARTDIR="$DEST.parts"; mkdir -p "$PARTDIR"
CHUNK=$(( (SIZE + NPARTS - 1) / NPARTS ))
log() { echo "[$(date '+%F %T')] $*"; }

fetch_part() {
  local i=$1 start=$(( $1 * CHUNK )) end=$(( ($1 + 1) * CHUNK - 1 ))
  (( end >= SIZE )) && end=$(( SIZE - 1 ))
  local want=$(( end - start + 1 )) f="$PARTDIR/part_$(printf %03d "$i")"
  for attempt in $(seq 1 20); do
    local have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
    (( have == want )) && return 0
    (( have > want )) && { rm -f "$f"; have=0; }
    curl -s -L -x "$PROXY" --max-time 1800 -r "$(( start + have ))-$end" "$URL" >> "$f"
  done
  log "part $i failed"; return 1
}
export -f fetch_part log; export URL PROXY CHUNK SIZE PARTDIR

log "start $DEST size=$SIZE parts=$NPARTS par=$PAR"
seq 0 $(( NPARTS - 1 )) | xargs -P "$PAR" -I{} bash -c 'fetch_part {}' || { log "some parts failed; re-run"; exit 1; }
cat "$PARTDIR"/part_* > "$DEST.tmp"
[ "$(stat -c %s "$DEST.tmp")" = "$SIZE" ] || { log "size mismatch"; exit 1; }
echo "$SHA  $DEST.tmp" | sha256sum -c - || { log "sha256 mismatch"; exit 1; }
mv "$DEST.tmp" "$DEST" && rm -rf "$PARTDIR"
log "done $DEST"
