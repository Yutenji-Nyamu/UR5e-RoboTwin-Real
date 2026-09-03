# Data management

[简体中文](zh-CN/DATA_MANAGEMENT.md)

The goal is to reuse valuable demonstrations while keeping every training and
evaluation result traceable to immutable source data.

## Lifecycle

1. **Raw**: append-only RTDE, gripper, image, sync, and session manifest output.
   Never edit a successful raw run in place.
2. **Validated**: attach timing/frame/action quality results and an acceptance or
   rejection reason.
3. **Converted**: create a versioned schema output from named raw run IDs. It is
   reproducible and may be regenerated.
4. **Training**: record dataset ID, policy, upstream commit, adapter commit,
   complete config, seed, environment lock, and checkpoint hashes.
5. **Evaluation**: link every trial to one checkpoint and dataset/config identity;
   store success, stop reason, latency, raw prediction, guarded command, and
   measured state.
6. **Archive**: retain valuable raw data and selected checkpoints; prune only
   reproducible caches and explicitly rejected/corrupt runs.

## Directory contract

```text
/data/robotics/ur5e-real/
├── raw/                 # collector-owned, append-only sessions
├── converted/           # schema-versioned datasets
├── checkpoints/         # policy/dataset/train-run hierarchy
└── logs/                # validation, training, shadow and live evaluation
```

Use a stable dataset ID such as
`pick_block_bowl__20260901__d001__schema-v1`. Its manifest lists source run IDs,
task/config, cameras/calibration, sample rate, schema version, converter commit,
quality result, and a checksum inventory. Human notes are additive; they do not
rename or mutate raw files.

## One trajectory, one index

Each recorded trajectory has one canonical `session_<run_id>.json`. Keep it
small: it points to products instead of repeating per-frame records already held
by the sync CSV. Before production collection, the collector should finalize
these fields when recording stops:

- identity: run ID, task type/name, schema version, and code commit;
- timing: start, finish, and duration;
- outcome: `unreviewed`, `success`, `failure`, or `aborted`, plus an optional
  stop/failure reason and short note;
- products: RTDE/action CSV, gripper-event CSV, sync CSV, head/wrist frame
  directories, and optional video/preview paths;
- counts: RTDE samples, gripper events, and synchronized frame pairs;
- provenance: robot, gripper, cameras, calibration ID, collection config, and
  sample rates.

The sync CSV remains the frame-level index. A later human or automated quality
review is an additive record keyed by run ID; it does not rewrite captured
sensor/action files. This single-index rule is also the boundary for future
RoboTwin conversion: converters consume accepted run IDs and produce a separate
dataset manifest.

## Retention

- Raw accepted demonstrations: keep by default.
- Rejected/corrupt raw runs: keep until the reason is reviewed, then archive or
  explicitly delete.
- Converted data: keep active versions; regenerate obsolete versions from raw.
- Checkpoints: keep `best`, `last`, and milestone checkpoints plus metrics;
  prune dense intermediate steps after evaluation.
- Caches and downloadable upstream assets: not authoritative; prune when space is
  needed.

## Moving a large existing directory

Never move an active Conda/venv/Git directory to the NTFS data disk. For datasets
or checkpoints:

1. confirm no process is using the source;
2. copy to `/data/robotics/staging` with preserved timestamps;
3. require a successful copy and a metadata dry run with no file-tree, size, or
   timestamp differences; use full content checksums only for irreplaceable data
   or suspected corruption;
4. rename the destination into its final shared location;
5. rename the source to a rollback name, create one explicit symlink, and run an
   application read/load test;
6. remove the rollback copy only after verification and record the migration.

This avoids training once and losing provenance, while still allowing large
artifacts to leave the NVMe.
