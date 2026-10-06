#!/usr/bin/env bash
# InternW0-Delta weights (RoboDojo post-trained + pretrain Base) and RynnBrain1.1-2B into the model zoo.
# Run on a training machine inside tmux; idempotent.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOO="${ZOO:-/ytech_milm_intern/danglingwei/model_zoos/InternW0-Delta}"
DL="$HERE/download_hf_chunked.sh"
export https_proxy=http://oversea-squid2.ko.txyun:11080 http_proxy=http://oversea-squid2.ko.txyun:11080

bash "$DL" InternRobotics/InternW0-Delta-RoboDojo robodojo.pt 12423419670 \
  344d4221d628462fb8d1f7263fdfeab74ed4701e7e0af25dde4a693b22968aaa "$ZOO/InternW0-Delta-RoboDojo" || exit 1
bash "$DL" Alibaba-DAMO-Academy/RynnBrain1.1-2B model.safetensors 4426558864 \
  2819d84441c27c18c403c1302f60ff1ba2c9cfcdc989657bf6589e8cd11a7e6d "$ZOO/RynnBrain1.1-2B" || exit 1
for f in .gitattributes README.md chat_template.jinja config.json generation_config.json processor_config.json \
         tokenizer.json tokenizer_config.json; do
  curl -sfL --retry 5 -o "$ZOO/RynnBrain1.1-2B/$f" "https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-2B/resolve/main/$f" \
    || { echo "FAILED RynnBrain $f"; exit 1; }
done
echo "[$(date '+%F %T')] RynnBrain small files OK"
bash "$DL" InternRobotics/InternW0-Delta-Base pretrain.pt 12423420182 \
  a6bcf7a1eb2cf3d303e02faca05579cf77f4ac90863a15ff896b7e85ef884a08 "$ZOO/InternW0-Delta-Base" || exit 1
echo "[$(date '+%F %T')] ALL DONE"
