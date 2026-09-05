# RoboTwin Diffusion Policy training

[简体中文](../zh-CN/runbooks/train.md)

Enter the shared environment and repository:

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
```

## 1. Build training data from successful sessions

```bash
ur5e-real convert --config configs/lab.yaml \
  --task pick_place_cube --task-config simple
```

Save the printed `HDF5_RUN` path; `N` is the number of successful episodes.
Convert it to RoboTwin DP Zarr:

```bash
ur5e-real process-dp HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N \
  --output /data/robotics/ur5e-real/converted/dp/<DATA_VERSION>.zarr \
  --trim-static-edges
```

The Zarr defaults to `/data/robotics/ur5e-real/converted/dp/`. It contains the
head camera, 14-dimensional TCP/gripper state, and
`action[t] = state[t+1]`. `--trim-static-edges` removes only contiguous idle
prefixes and suffixes using 2 mm/1 degree thresholds while retaining three
context frames at each edge; interior pauses remain. Source run IDs, original
lengths, and exact trim bounds are stored in Zarr attributes. Put the episode
count, trim rule, and HDF5 timestamp in the filename and avoid `--overwrite`,
so pre- and post-trim variants coexist.

## 2. Smoke training

```bash
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N \
  --debug --batch-size 8
```

`--debug` retains the RoboTwin model but limits training to two epochs and at
most three steps per epoch. Validation is automatically disabled for a
single-episode dataset.

## 3. Full training

```bash
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N
```

Without `--debug`, this preserves RoboTwin `robot_dp_14.yaml`: horizon 8,
three observations, six actions, 10 Hz, and 600 epochs. Batch size is capped at
128 and automatically fits within the shortest episode for small datasets so
the held-out episode produces at least one validation batch. Run output and
checkpoints are written under `/data/robotics/ur5e-real/checkpoints/dp/`.

The default saves every 300 epochs, at epochs 300 and 600. Per-epoch train and
validation losses are written to `logs.json.txt` in the run directory.

The first full run on 2026-09-04 included six successful episodes (869
transitions), held out the last episode for validation, and trained on the
other five. All 600 epochs finished in about 15.6 minutes; both `300.ckpt` and
`600.ckpt` were produced and loaded successfully against the held-out episode.
Training loss converged, while validation plateaued after roughly 300 epochs
and rose slightly near the end, so both checkpoints remain for live comparison.
Two aborted sessions remain in raw storage but were excluded from this dataset.

The 2026-09-05 version contains 17 successful episodes from
`run_20260905_150058`, reducing 2109 to 1562 transitions with the edge rule
above. Training directories carry timestamps; select checkpoints uniquely with
`<TRAIN_TIMESTAMP>:300` or `<TRAIN_TIMESTAMP>:600`. Brief data changes are
recorded in `/data/robotics/ur5e-real/DATA_LOG.md`.

## ACT (retained adapter)

```bash
ur5e-real process-act HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N
scripts/train_act.sh pick_place_cube simple N 0 0
```

ACT and DP remain parallel adapters; DP is the current main path.
