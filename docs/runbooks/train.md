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
  --task pick_place_cube --task-config simple --episodes N
```

The Zarr defaults to `/data/robotics/ur5e-real/converted/dp/`. It contains the
head camera, 14-dimensional TCP/gripper state, and
`action[t] = state[t+1]`.

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

## ACT (retained adapter)

```bash
ur5e-real process-act HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N
scripts/train_act.sh pick_place_cube simple N 0 0
```

ACT and DP remain parallel adapters; DP is the current main path.
