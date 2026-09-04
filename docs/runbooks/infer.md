# RoboTwin DP inference

[简体中文](../zh-CN/runbooks/infer.md)

Enter the shared environment and repository:

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
ur5e-real infer-dp --checkpoint CHECKPOINT \
  --config configs/lab.yaml --shadow --chunks 10
```

This connects both cameras and read-only RTDE, prints predictions, and sends no
robot or gripper command. `--chunks 0` runs until `Ctrl+C`. The real chain has
passed two chunks; measured steady inference was about 0.84 seconds.

## 3. Live execution

Initialize under PolyScope **Remote Control**:

```bash
ur5e-policy-init
```

Then switch PolyScope to **Local**, start the robot-side program built from
`robot_programs/servoj_control_loop.script`, and execute one chunk:

```bash
ur5e-real infer-dp --checkpoint CHECKPOINT \
  --config configs/lab.yaml --execute --chunks 1
```

Each inference preserves and executes all six native RoboTwin actions. Every
10 Hz TCP target is interpolated into the 500 Hz RTDE servoJ stream. The gripper
uses element 14; add `--no-gripper` to disable it temporarily.

Offline and shadow modes are hardware-tested. The `--execute` path is
implemented; the next on-site step is a servoJ hold and millimeter-scale move
before attaching a fully trained model.
