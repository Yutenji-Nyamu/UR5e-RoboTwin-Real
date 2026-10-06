#!/usr/bin/env bash
# Node-local VLABench install on a fresh machine: assets (git part + obj.zip / scene.zip) unpacked to local disk at
# /usr/local/vlabench/assets (the target of the repo's VLABench/assets symlink), then the venv.
#   tmux new-session -d -s vlab_env "bash scripts/vlabench/setup_node_local.sh 2>&1 | tee /tmp/vlab_env.log"
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/ytech_milm_intern/danglingwei/files/VLABench/VLABench
ZIP_DIR=/ytech_milm_intern/danglingwei/datas/VLABench/assets_zip
LOCAL=/usr/local/vlabench/assets
mkdir -p "$LOCAL"
cp -a "$ROOT/assets_git/." "$LOCAL/"
(cd "$LOCAL" && unzip -q -o "$ZIP_DIR/scene.zip" && unzip -q -o "$ZIP_DIR/obj.zip")
echo "assets: $(du -sh "$LOCAL" | cut -f1)"
bash scripts/vlabench/setup_vlabench_env.sh
