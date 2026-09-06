from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"[RUN] {' '.join(command)}")
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def _fstab_entry(mountpoint: Path) -> tuple[str, str]:
    result = _run(
        [
            "findmnt",
            "--fstab",
            "--noheadings",
            "--output",
            "SOURCE,FSTYPE",
            "--target",
            str(mountpoint),
        ],
        capture=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != 2:
        raise RuntimeError(f"expected one fstab entry for {mountpoint}")
    return fields[0], fields[1]


def _device_path(source: str) -> Path:
    prefixes = {
        "UUID=": "/dev/disk/by-uuid/",
        "LABEL=": "/dev/disk/by-label/",
        "PARTUUID=": "/dev/disk/by-partuuid/",
        "PARTLABEL=": "/dev/disk/by-partlabel/",
    }
    for prefix, directory in prefixes.items():
        if source.startswith(prefix):
            return Path(directory) / source.removeprefix(prefix)
    path = Path(source)
    if not path.is_absolute():
        raise RuntimeError(f"unsupported fstab source: {source}")
    return path


def _systemd_unit(mountpoint: Path, suffix: str) -> str:
    result = _run(
        ["systemd-escape", "--path", f"--suffix={suffix}", str(mountpoint)],
        capture=True,
    )
    return result.stdout.strip()


def _mounted_ntfs_options(mountpoint: Path) -> set[str]:
    result = _run(
        ["findmnt", "--target", str(mountpoint), "--noheadings", "--output", "FSTYPE,OPTIONS"],
        capture=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if fields and fields[0] in {"ntfs", "ntfs3", "fuseblk"}:
            return set(fields[1].split(",")) if len(fields) == 2 else set()
    raise RuntimeError(f"no NTFS filesystem mounted below {mountpoint}")


def repair(mountpoint: Path) -> None:
    mountpoint = mountpoint.resolve()
    if mountpoint == Path("/"):
        raise RuntimeError("refusing to operate on the filesystem root")

    source, fstype = _fstab_entry(mountpoint)
    if fstype not in {"ntfs", "ntfs3", "fuseblk"}:
        raise RuntimeError(f"{mountpoint} is configured as {fstype}, not NTFS")

    device = _device_path(source)
    mount_unit = _systemd_unit(mountpoint, "mount")
    automount_unit = _systemd_unit(mountpoint, "automount")

    print(f"[STORAGE] {device} -> {mountpoint}")
    try:
        _run(["systemctl", "stop", mount_unit, automount_unit])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"could not unmount {mountpoint}; close terminals and programs using it, "
            f"then retry (inspect with: fuser -vm {mountpoint})"
        ) from exc

    _run(["ntfsfix", "--clear-dirty", str(device)])
    _run(["systemctl", "daemon-reload"])
    subprocess.run(
        ["systemctl", "reset-failed", mount_unit, automount_unit],
        check=False,
        text=True,
    )
    _run(["systemctl", "start", automount_unit])

    # Accessing an x-systemd.automount path triggers the real filesystem mount.
    with os.scandir(mountpoint) as entries:
        next(entries, None)
    options = _mounted_ntfs_options(mountpoint)
    if "rw" not in options:
        raise RuntimeError(f"{mountpoint} mounted without rw: {','.join(sorted(options))}")
    print(f"[READY] {mountpoint} mounted read-write")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ur5e-storage-repair",
        description="repair and remount the configured NTFS data disk",
    )
    parser.add_argument("--mountpoint", type=Path, default=Path("/data"))
    args = parser.parse_args()

    if os.geteuid() != 0:
        os.execvp(
            "sudo",
            ["sudo", sys.executable, "-m", "ur5e_real.storage", "--mountpoint", str(args.mountpoint)],
        )

    try:
        repair(args.mountpoint)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
