# Workstation storage

[简体中文](zh-CN/STORAGE.md)

Inventory updated: 2026-09-12, around 15:28 CST, during training. No cleanup or migration was performed.

## Active layout

| Device | Filesystem | State | Role |
|---|---|---|---|
| 1 TB Lenovo NVMe | ext4 | 937 GiB total, 531 GiB used (60%), 359 GiB available | OS, source, Git, Conda/venv, current π0.5 checkpoints |
| 4 TB Seagate HDD | NTFS3 | `/data` mounted rw, 736 GiB used (20%), about 3.0 TiB available | datasets, recordings, DP checkpoints, archives |

## System-disk consumers and growth

| Readable area | Approximate size | Breakdown |
| --- | ---: | --- |
| Current project | 100 GiB | π0.5 checkpoints 70 GiB; environments/caches 30 GiB; source and small logs are minor |
| Current user's Conda | 56 GiB | environments 48 GiB, pkgs cache 3.7 GiB, base/tools |
| `ScutPythonProject` | 37 GiB | RoboTwin 25 GiB, DexVLA 5.3 GiB, LLMs 4.4 GiB |
| Legacy `UR5e_DataCollection` | 33 GiB | RoboTwin 24 GiB, camera data 4.9 GiB, old Git history 3.5 GiB |
| `/usr`, `/var`, `/opt` | 22, 16, 3.5 GiB | OS/software; `/var` includes CUDA installer repos 6.3 GiB and logs 1.9 GiB |

Old user directories and some system directories were unreadable: this is not a fully reconciled disk ledger.
Unreadable directories are not zero-sized. `du` deduplicates hard links, so cache/environment subtrees
cannot be summed as independently reclaimable storage.

Checkpoints are the main ongoing growth source: approximately 8.8 GiB each. Three completed π0.5 runs
retain steps 500/1000 (about 53 GiB together); the batch=2 three-step probe retains another 8.8 GiB.
The charger run has saved step 500 (8.8 GiB); steps 1000/1500 will add about 17.6 GiB,
leaving approximately 341 GiB available. `.venv` also contains base weights 12 GiB, uv cache 13 GiB,
dummy-test checkpoints 3.1 GiB and HF dataset cache 2.1 GiB.

There is no imminent capacity shortage. Long term, archive inactive checkpoints to `/data`, starting
with expendable probes/dummy artifacts while retaining useful trained models. Do migration/cleanup
as a separate verified operation; do not move active model/environment paths during training.
This inspection only archived small evidence into Git and did not delete or migrate any data.

The HDD is persistently configured by UUID in `/etc/fstab` with `nofail` and a
systemd automount. On 2026-09-06 the repository's `ur5e-storage-repair`
repaired the MFT mirror, cleared the dirty flag, and verified a fresh `rw`
mount. Accounts `zhangw`, `ur5`, and `wlf` belong to `robotdata`;
`/data/robotics` is setgid group-writable. A pre-change fstab backup is retained
at `/etc/fstab.codex-backup-20260901`.

NTFS keeps the existing cross-platform data intact, but source trees, Conda,
venv, and workloads that depend heavily on Unix links/permissions stay on ext4.

## Large-data migration completed

DETwinVLA (208.1 GiB), DexVLA (89.5 GiB), Pi0.5 checkpoints and training data
(about 74.8 GiB), ACT data and models (about 21 GiB), and large download
archives/model shards (about 17.4 GiB) now live under
`/data/robotics/shared`. Symlinks remain at their old locations, so existing
commands keep working. That migration's capacity figures were historical; use the current table above.

Re-downloadable pip/Conda caches should be cleaned instead. Conda environments,
Git worktrees, and caches that create many small files stay off NTFS.

## Performance boundary

The 4 TB device is a mechanical disk. It is well suited to sequential recording,
HDF5, video, checkpoint storage, and archives, but random access to many small
PNG files is much slower than NVMe. Initial model loading is also slower; after
weights are in RAM/GPU, the impact is usually small. Raw data and DP checkpoints are on `/data`;
current π0.5 checkpoints are still in the repository's `checkpoints/` on NVMe. Do not confuse a future
archive recommendation with a completed migration. Use HDF5 or shards for training input. Stage only
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

## One-command read-only mount repair

An unclean NTFS volume may cause Linux to mount `/data` read-only. First leave
shells whose working directory is under `/data` and stop collection or training
that uses the disk, then run:

```bash
ur5e-storage-repair
```

The command derives the device UUID and systemd units from `/etc/fstab`, stops
the mount normally, runs `ntfsfix --clear-dirty`, restores the automount, and verifies that
the underlying NTFS mount is `rw`. It requests system privileges once through
`sudo`, does not rewrite fstab, and never uses forced or lazy unmounting. If the
disk is busy, use `fuser -vm /data`, exit the listed programs normally, and
retry. If `ntfsfix` reports Windows hibernation or explicitly requires
`chkdsk`, boot Windows, check the volume, and shut it down fully instead of
forcing the Linux mount.
