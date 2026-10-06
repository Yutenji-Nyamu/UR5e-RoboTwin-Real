#!/usr/bin/env bash
# Build third_party/RoboTwin as a symlink overlay over the predecessor's RoboTwin 2.0 checkout.
# Only the two curobo planner yml files are copied and rewritten: they hard-code absolute paths of a
# deleted checkout (/m2v_intern2/.../XWAM_260716), which breaks `load_robot` at replay time.
set -euo pipefail
SRC=/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824/third_party/RoboTwin
ASSETS=/m2v_intern_v3/danglingwei/files/RoboTwin_260625/assets/embodiments
DST="$(cd "$(dirname "$0")/../../.." && pwd)/third_party/RoboTwin"
mkdir -p "$DST/assets/embodiments" "$DST/data"
for entry in envs script task_config description code_gen policy LICENSE README.md; do
  ln -sfn "$SRC/$entry" "$DST/$entry"
done
for entry in _download.py files background_texture objects; do
  ln -sfn "$(readlink -f "$SRC/assets/$entry")" "$DST/assets/$entry"
done
for emb in "$ASSETS"/*/; do
  name="$(basename "$emb")"
  if [[ "$name" != "aloha-agilex" ]]; then
    ln -sfn "$emb" "$DST/assets/embodiments/$name"
  fi
done
mkdir -p "$DST/assets/embodiments/aloha-agilex"
for f in "$ASSETS"/aloha-agilex/*; do
  base="$(basename "$f")"
  case "$base" in
    curobo_left.yml|curobo_right.yml)
      sed "s#/m2v_intern2/danglingwei/codes/XWAM_260716/third_party/RoboTwin#$DST#g" "$f" > "$DST/assets/embodiments/aloha-agilex/$base" ;;
    *) ln -sfn "$f" "$DST/assets/embodiments/aloha-agilex/$base" ;;
  esac
done
echo "overlay at $DST"
grep -n "urdf_path\|collision_spheres" "$DST/assets/embodiments/aloha-agilex/curobo_left.yml"
