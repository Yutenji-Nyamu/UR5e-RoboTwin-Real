"""Verify a read-only local pi05_base copy against official GCS sizes and CRC32C."""

import argparse
import base64
import json
from pathlib import Path
import urllib.request

import google_crc32c

from ur5e_real.adapters.robotwin_pi05.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    prefix = "checkpoints/pi05_base/params/"
    url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=" + prefix
    with urllib.request.urlopen(url, timeout=30) as response:
        listing = json.load(response)
    if listing.get("nextPageToken") or not listing.get("items"):
        raise ValueError("expected a complete, nonempty official object listing")
    expected = {item["name"].removeprefix(prefix): item for item in listing["items"]}
    actual = {str(path.relative_to(args.params)) for path in args.params.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ValueError(f"cache files differ: missing={set(expected) - actual}; extra={actual - set(expected)}")
    verified = []
    for relative, item in sorted(expected.items()):
        path = args.params / relative
        if path.stat().st_size != int(item["size"]):
            raise ValueError(f"incorrect size: {relative}")
        checksum = google_crc32c.Checksum()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        crc = base64.b64encode(checksum.digest()).decode()
        if crc != item["crc32c"]:
            raise ValueError(f"incorrect CRC32C: {relative}")
        verified.append({"path": relative, "size": int(item["size"]), "generation": item["generation"], "crc32c": crc})
        print(f"[VERIFIED] {relative}: {item['size']} bytes", flush=True)
    report = {
        "source": "gs://openpi-assets/checkpoints/pi05_base/params",
        "local_params": str(args.params.resolve()),
        "verification": "all_sizes_and_crc32c_match_official_gcs",
        "files": verified,
        "total_bytes": sum(item["size"] for item in verified),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
