# RoboTwin DP inference

[简体中文](../zh-CN/runbooks/infer.md)

For low-level commands, first enter the shared environment and repository:

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
```

## 1. Offline inference

```bash
ur5e-real infer-dp --checkpoint CHECKPOINT \
  --episode HDF5_EPISODE --index 20 --output prediction.npz
```

This reads recorded images and state, then reports a six-action prediction,
latency, and label error without connecting hardware.

## 2. Live shadow mode

```bash
ur5e-infer <TRAIN_TIMESTAMP>:600 --shadow --chunks 10
```

This connects both cameras and read-only RTDE, prints predictions, and sends no
robot or gripper command. `--chunks 0` runs until `Ctrl+C`. Pass a full path or
`<TRAIN_TIMESTAMP>:<EPOCH>` so same-epoch checkpoints remain unambiguous.

## 3. Live execution

Keep PolyScope in **Remote Control**. Initialization checks devices, returns to
home at low speed, and opens the gripper:

```bash
ur5e-infer-init
```

After arranging the scene, first test the socket path inherited from the
historically successful ACT deployment. It reads RTDE feedback and sends
`speedl` over socket 30001; no PolyScope RTDE program is required:

```bash
ur5e-infer <TRAIN_TIMESTAMP>:600 --execute \
  --backend socket --smooth-alpha 0.7 --chunks 1 --no-gripper
```

After confirming motion, run the complete episode with the gripper enabled:

```bash
ur5e-infer <TRAIN_TIMESTAMP>:600 --execute \
  --backend socket --smooth-alpha 0.7
```

`--smooth-alpha 0.7` applies the target EMA used by the old ACT path; `1.0`
disables it, while smaller values trade response for more smoothing. The default
50 chunks equal RoboTwin's original 300-action limit. `Ctrl+C` stops early.

The RTDE servoJ implementation remains an explicit experimental alternative:

```bash
ur5e-infer <TRAIN_TIMESTAMP>:600 --execute --backend rtde --chunks 1 --no-gripper
```

It injects the robot-side servoJ script and writes RTDE input registers. Script
runtime has been confirmed, but known-displacement tracking is not yet proven.
The backends never switch automatically. The gripper uses action element 14;
`--no-gripper` isolates arm tests. Camera and RTDE state acquisition run in
background threads so inference does not leave stale samples queued.
