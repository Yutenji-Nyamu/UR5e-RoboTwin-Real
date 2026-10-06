"""Download OpenWAM-Alpha-Sim-VLABench and OpenWAM's LeRobot repack of the VLABench primitive fine-tuning data.

    HF_HUB_DISABLE_XET=1 https_proxy=http://oversea-squid1.jp.txyun:11080 \
        /usr/bin/python3.10 scripts/vlabench/download_vlabench.py --what ckpt data
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

ROOT = "/ytech_milm_intern/danglingwei/datas/VLABench"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", nargs="+", choices=["ckpt", "data"], default=["ckpt", "data"])
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    if "data" in args.what:
        print("data ->", snapshot_download(repo_id="OpenWAM/VLABench", repo_type="dataset",
                                           local_dir=f"{ROOT}/OpenWAM-VLABench-tars", max_workers=args.workers),
              flush=True)
    if "ckpt" in args.what:
        print("ckpt ->", snapshot_download(repo_id="OpenWAM/OpenWAM-Alpha-Sim-VLABench",
                                           local_dir=f"{ROOT}/OpenWAM-Alpha-Sim-VLABench", max_workers=args.workers),
              flush=True)


if __name__ == "__main__":
    main()
