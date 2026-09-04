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
ur5e-infer 600 --shadow --chunks 10
```

This connects both cameras and read-only RTDE, prints predictions, and sends no
robot or gripper command. `--chunks 0` runs until `Ctrl+C`. `600` resolves to
the only current `600.ckpt`; a full path is also accepted. The real chain has
passed two chunks; measured steady inference was about 0.84 seconds.

## 3. Live execution

Keep PolyScope in **Remote Control**. Initialization checks devices, returns to
home at low speed, and opens the gripper:

```bash
ur5e-infer-init
```

After arranging the scene, execute one chunk for the first commissioning run:

```bash
ur5e-infer 600 --execute --chunks 1
```

After servoJ is confirmed, run a complete episode with:

```bash
ur5e-infer 600 --execute
```

This command selects the project environment, directory, `configs/lab.yaml`,
and RoboTwin path. A socket sends `servoj_control_loop.script` once; all state
feedback and action targets then use RTDE. The default 50 chunks equal
RoboTwin's original 300-action limit, and `Ctrl+C` stops early.

Each inference preserves and executes all six native RoboTwin actions. Every
10 Hz TCP target is interpolated into the 500 Hz RTDE servoJ stream. The gripper
uses element 14; add `--no-gripper` to disable it temporarily.

Offline and shadow modes are hardware-tested. Execute is wired; the next live
session first uses `--chunks 1` to verify automatic program startup, servoJ hold,
and small motion before running a complete episode.
