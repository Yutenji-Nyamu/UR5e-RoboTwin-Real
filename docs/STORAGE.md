# Workstation storage

[简体中文](zh-CN/STORAGE.md)

Inventory updated: 2026-09-03.

## Active layout

| Device | Filesystem | State | Role |
|---|---|---|---|
| 1 TB Lenovo NVMe | ext4 | 937 GiB usable, 428 GiB used, 462 GiB available | OS, source, Git, Conda/venv, active small files |
| 4 TB Seagate HDD | NTFS3 | mounted at `/data`, 709 GiB used, 3.0 TiB available | datasets, recordings, checkpoints, archives |

The HDD is persistently configured by UUID in `/etc/fstab` with `nofail` and a
systemd automount. Accounts `zhangw`, `ur5`, and `wlf` belong to `robotdata`;
`/data/robotics` is setgid group-writable. A pre-change fstab backup is retained
at `/etc/fstab.codex-backup-20260901`.

NTFS keeps the existing cross-platform data intact, but source trees, Conda,
venv, and workloads that depend heavily on Unix links/permissions stay on ext4.

## Large-data migration completed

DETwinVLA (208.1 GiB), DexVLA (89.5 GiB), Pi0.5 checkpoints and training data
(about 74.8 GiB), ACT data and models (about 21 GiB), and large download
archives/model shards (about 17.4 GiB) now live under
`/data/robotics/shared`. Symlinks remain at their old locations, so existing
commands keep working; the NVMe now has about 462 GiB available.

Re-downloadable pip/Conda caches should be cleaned instead. Conda environments,
Git worktrees, and caches that create many small files stay off NTFS.

## Performance boundary

The 4 TB device is a mechanical disk. It is well suited to sequential recording,
HDF5, video, checkpoint storage, and archives, but random access to many small
PNG files is much slower than NVMe. Initial model loading is also slower; after
weights are in RAM/GPU, the impact is usually small. Keep raw data and
checkpoints on `/data` and convert training input to HDF5 or shards. Stage only
the active dataset on NVMe if measurements show an actual training I/O
bottleneck, then move it back after the run.

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
