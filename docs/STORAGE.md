# Workstation storage

[简体中文](zh-CN/STORAGE.md)

Inventory: 2026-09-01. No historical project or model data was deleted.

## Active layout

| Device | Filesystem | State | Role |
|---|---|---|---|
| 1 TB Lenovo NVMe | ext4 | 937 GiB usable, 839 GiB used, 52 GiB available | OS, source, Git, Conda/venv, active small files |
| 4 TB Seagate HDD | NTFS3 | mounted at `/data`, 299 GiB used, 3.4 TiB available | datasets, recordings, checkpoints, archives |

The HDD is persistently configured by UUID in `/etc/fstab` with `nofail` and a
systemd automount. Accounts `zhangw`, `ur5`, and `wlf` belong to `robotdata`;
`/data/robotics` is setgid group-writable. A pre-change fstab backup is retained
at `/etc/fstab.codex-backup-20260901`.

NTFS keeps the existing cross-platform data intact, but source trees, Conda,
venv, and workloads that depend heavily on Unix links/permissions stay on ext4.

## Root-disk ownership

| Account/area | Approx. size | Dominant content |
|---|---:|---|
| `zhangw` account | 480 GiB | `ScutPythonProject` 354.7 GiB, mostly one RoboTwin policy tree |
| `ur5` account | 272.4 GiB | RoboTwin/Pi0.5 work 130.2 GiB, caches 90.5 GiB, Anaconda 45.3 GiB |
| `wlf` account | 38.2 GiB | caches 31.9 GiB, mostly pip and Hugging Face |

Largest regular-data migration candidates:

1. old DETwinVLA checkpoints: about 208.1 GiB;
2. DexVLA checkpoints: about 89.5 GiB;
3. Pi0.5 checkpoints and training data: about 59.3 + 15.5 GiB;
4. historical RoboTwin ACT/processed datasets: tens of GiB.
5. three top-level download archives/model shards: about 17.4 GiB total.

These can move to `/data/robotics/shared` and be linked back after copy and
checksum verification. Current GPU/process inspection found no active training,
but migration is still scheduled as an explicit I/O job rather than performed
during hardware commissioning.

The two largest checkpoint directories contain no internal symlinks, making
them the safest high-impact candidates; moving both would recover about
297.6 GiB on the NVMe.

Pip caches (about 30.5 GiB across two accounts) and Conda package caches are
re-downloadable and should normally be cleaned, not archived. Hugging Face
caches, full environments, and Git worktrees are not blindly moved to NTFS.

## Data-disk directories

```text
/data/robotics/
├── ur5e-real/{raw,converted,checkpoints,logs}
├── shared/{datasets,models,checkpoints,archives}
└── staging
```

The local `configs/lab.yaml` uses `/data/robotics/ur5e-real` as
`collection.data_root`. See [`DATA_MANAGEMENT.md`](DATA_MANAGEMENT.md) for
dataset identity, retention, and verified migration rules.
