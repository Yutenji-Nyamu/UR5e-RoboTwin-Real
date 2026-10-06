"""Download the parts of EBench needed for local evaluation and the OpenWAM-Alpha EBench checkpoint.

Assets: shared scene/object/robot USDs plus the pre-generated task instances of the requested splits
(``<task>_<split>/<idx>/``). The un-suffixed task instances (demo-generation layouts) and the
LeRobot training dataset are not downloaded here.

    HF_HUB_DISABLE_XET=1 https_proxy=http://oversea-squid1.jp.txyun:11080 \
        /usr/bin/python3.10 scripts/ebench/download_ebench.py --what assets ckpt
"""

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

ROOT = "/ytech_milm_intern/danglingwei/datas/EBench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", nargs="+", choices=["assets", "ckpt"], default=["assets", "ckpt"])
    parser.add_argument("--splits", nargs="+", default=["val_unseen", "test_mini"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if "ckpt" in args.what:
        path = snapshot_download(
            repo_id="OpenWAM/OpenWAM-Alpha-Sim-EBench",
            local_dir=f"{ROOT}/OpenWAM-Alpha-Sim-EBench",
            max_workers=args.workers,
        )
        print("ckpt ->", path, flush=True)

    if "assets" in args.what:
        patterns = ["assets/**"]
        for split in args.splits:
            patterns += [f"tasks/ebench/*/*_{split}/**"]
        path = snapshot_download(
            repo_id="InternRobotics/EBench-Assets",
            repo_type="dataset",
            local_dir=f"{ROOT}/EBench-Assets",
            allow_patterns=patterns,
            max_workers=args.workers,
        )
        print("assets ->", path, flush=True)


if __name__ == "__main__":
    main()
