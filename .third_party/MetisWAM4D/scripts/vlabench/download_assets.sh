#!/usr/bin/env bash
# VLABench scene/object assets (same obj.zip / scene.zip as the official Google Drive links, which hit gdown quota).
set -euo pipefail
export https_proxy=${PROXY:-http://oversea-squid1.jp.txyun:11080} http_proxy=${PROXY:-http://oversea-squid1.jp.txyun:11080}
export HF_HUB_DISABLE_XET=1
ZIP_DIR=${ZIP_DIR:-/ytech_milm_intern/danglingwei/datas/VLABench/assets_zip}
ASSETS=${ASSETS:-/ytech_milm_intern/danglingwei/files/VLABench/VLABench/assets}

/usr/bin/python3.10 - "$ZIP_DIR" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
for f in ("scene.zip", "obj.zip"):
    print(hf_hub_download("JianZhangAI/VLABench_Install", f, repo_type="dataset", local_dir=sys.argv[1]), flush=True)
EOF
cd "$ASSETS"
unzip -q -o "$ZIP_DIR/scene.zip"
unzip -q -o "$ZIP_DIR/obj.zip"
ls "$ASSETS"
