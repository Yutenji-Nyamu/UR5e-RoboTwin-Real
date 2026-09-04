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
three observations, six actions, 10 Hz, batch 128, and 600 epochs. Run output
and checkpoints are written under `/data/robotics/ur5e-real/checkpoints/dp/`.

On 2026-09-04 one real episode passed the native Dataset read, two-epoch GPU
training, and production of two reloadable checkpoints.

## ACT (retained adapter)

```bash
ur5e-real process-act HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N
scripts/train_act.sh pick_place_cube simple N 0 0
```

ACT and DP remain parallel adapters; DP is the current main path.
