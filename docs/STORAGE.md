# Workstation storage layout

Inventory snapshot: 2026-09-01. Sizes are approximate and no data was deleted.

## Physical disks

| Device | Usable size | Current format/use |
|---|---:|---|
| 1 TB Lenovo NVMe (`nvme0n1`) | 953.9 GiB | Ubuntu root filesystem; code, home directories, active environments |
| 4 TB Seagate HDD (`sda`) | 3.64 TiB | One NTFS data partition; about 299 GiB used and 3.4 TiB free |

The 4 TB disk is real and healthy enough to read. It has no `/etc/fstab` entry,
so Linux normally leaves it unmounted; this is why the nearly full root disk did
not benefit from its free space. NTFS is suitable for large datasets and archive
files, but Linux Conda environments and active Git trees should remain on ext4
because they rely on Unix permissions, links, and many small-file operations.

The largest existing areas on the 4 TB partition are:

- Hugging Face LeRobot/parquet caches: about 199.5 GiB;
- `wlf_data`: about 72.4 GiB, including another RoboTwin tree and Anaconda;
- simulator data: about 16.3 GiB;
- a `pi05_move_can_pot_400` dataset: about 8.8 GiB.

## Secondary robotics account on the NVMe

The `ur5` account uses about 272.4 GiB. It is mostly a coherent
RoboTwin/Pi0.5/OpenVLA workspace rather than miscellaneous personal files:

| Area | Approx. size | Main contents |
|---|---:|---|
| `work` | 130.2 GiB | `RoboTwin-pi05` (114.4 GiB) plus an older RoboTwin tree |
| `work/RoboTwin-pi05/policy` | 84.4 GiB | Pi0.5 checkpoints (~59.3 GiB), training data (~15.5 GiB), `.venv` (~9.6 GiB) |
| `.cache` | 90.5 GiB | Hugging Face models (~65.5 GiB), pip (~13.0 GiB), OpenPI (~11.6 GiB) |
| `anaconda3` | 45.3 GiB | environments (~29.4 GiB) and package cache (~11.5 GiB) |

The model checkpoints and datasets are likely valuable experiments. Package and
pip caches are more disposable, but they should not be removed during a general
cleanup without first deciding whether fast environment restoration matters.

## Placement policy

- NVMe: source code, Conda environments, active small checkpoints, temporary
  build files.
- 4 TB disk: raw recordings, converted datasets, full checkpoints, model caches,
  old RoboTwin experiments, and archives.
- Configure `collection.data_root` as a path on the mounted 4 TB disk, for example
  `/data/ur5e_real`, before the next recording.
- Migrate large directories by copy, checksum, switch configuration, verify, and
  only then remove the old copy. Do not use blind symlinks or mass deletion.

Short term, mount the existing NTFS partition persistently at `/data`. Long term,
after a verified backup, an ext4 data partition would give more predictable Linux
behaviour; repartitioning is destructive and is intentionally outside the current
inspection.
