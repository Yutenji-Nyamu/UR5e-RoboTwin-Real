#!/usr/bin/env bash
# Unpack IG-10K ManiSkill assets onto this machine's local disk (~/.maniskill/data) and link the local RoboTwin assets.
set -euo pipefail
SRC=/ytech_milm_intern/danglingwei/datas/IG-10K-Assets
DST=${MS_ASSET_DIR:-$HOME/.maniskill}/data
RT=/m2v_intern_v3/danglingwei/files/RoboTwin_260625/assets
mkdir -p "$DST/robotwin"
cd "$DST"
for f in assets sketchfab partnet_mobility; do
  [ -e "$DST/.done_$f" ] && continue
  /usr/bin/python3.10 - "$SRC/$f.tar.zst" <<'EOF'
import sys, tarfile, zstandard
with open(sys.argv[1], "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r, tarfile.open(fileobj=r, mode="r|") as t:
    n = 0
    for m in t:
        t.extract(m, ".")
        n += 1
print(sys.argv[1], "members", n)
EOF
  touch "$DST/.done_$f"
done
ln -sfn "$RT/objects" "$DST/robotwin/objects"
ln -sfn "$RT/background_texture" "$DST/robotwin/background_texture"
du -sh "$DST"/* | sed "s|$DST/||"
echo "assets ready on $(hostname)"
