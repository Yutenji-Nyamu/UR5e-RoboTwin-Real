"""Download one shard of EBench-Dataset (manifest built from the HF file list, top camera excluded).

ModelScope through the internal proxy is the fastest source here (~20 MB/s per machine, additive across machines);
hf-mirror through the same proxy is the fallback. Shards are whole task directories balanced by size, so machines
never write the same file; files already present with the manifest size are skipped, so reruns resume.

    /usr/bin/python3.10 scripts/ebench/download_dataset.py --shard 0 --num-shards 3 --workers 32
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = "/ytech_milm_intern/danglingwei/datas/EBench"
MANIFEST = f"{ROOT}/EBench-Dataset_manifest.json"
DEST = f"{ROOT}/EBench-Dataset"
PROXIES = {"http": "http://10.66.29.113:11080", "https": "http://10.66.29.113:11080"}
SOURCES = [
    "https://www.modelscope.cn/api/v1/datasets/InternRobotics/EBench-Dataset/repo?Revision=master&FilePath={path}",
    "https://hf-mirror.com/datasets/InternRobotics/EBench-Dataset/resolve/{sha}/{path}",
]


def shard_files(files, shard, num_shards):
    by_task = defaultdict(list)
    for f in files:
        by_task["/".join(f["path"].split("/")[:2])].append(f)
    loads = [0] * num_shards
    owner = {}
    for task, fs in sorted(by_task.items(), key=lambda kv: -sum(f["size"] for f in kv[1])):
        i = loads.index(min(loads))
        owner[task] = i
        loads[i] += sum(f["size"] for f in fs)
    return [f for task, fs in by_task.items() if owner[task] == shard for f in fs]


def fetch(f, sha, session):
    out = os.path.join(DEST, f["path"])
    if os.path.exists(out) and os.path.getsize(out) == f["size"]:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    last = None
    for attempt in range(6):
        url = SOURCES[min(attempt // 2, 1)].format(path=f["path"], sha=sha)
        try:
            with session.get(url, proxies=PROXIES, timeout=300, stream=True) as r:
                r.raise_for_status()
                tmp = out + ".part"
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            if os.path.getsize(tmp) != f["size"]:
                raise IOError(f"size {os.path.getsize(tmp)} != {f['size']}")
            os.replace(tmp, out)
            return f["size"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{f['path']}: {last}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    manifest = json.load(open(MANIFEST))
    files = shard_files(manifest["files"], args.shard, args.num_shards)
    total = sum(f["size"] for f in files)
    print(f"shard {args.shard}/{args.num_shards}: {len(files)} files, {total / 1e9:.1f} GB", flush=True)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)
    done, failed, t0 = 0, [], time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, f, manifest["sha"], session): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                done += fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append(str(e))
            if i % 200 == 0 or i == len(files):
                dt = time.time() - t0
                print(f"{i}/{len(files)} files, {done / 1e9:.1f} GB new, {done / 1e6 / dt:.1f} MB/s, "
                      f"{len(failed)} failed", flush=True)
    for e in failed:
        print("FAILED", e, flush=True)


if __name__ == "__main__":
    main()
