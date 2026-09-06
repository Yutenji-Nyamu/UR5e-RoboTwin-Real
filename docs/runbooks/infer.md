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
ur5e-infer 20260905_150221:600 --shadow --chunks 10
```

This connects both cameras and read-only RTDE, prints predictions, and sends no
robot or gripper command. `--chunks 0` runs until `Ctrl+C`. Pass a full path or
`TRAIN_TIMESTAMP:EPOCH` so same-epoch checkpoints remain unambiguous.

## 3. Live execution

Keep PolyScope in **Remote Control**. Initialization checks devices, returns to
home at low speed, and opens the gripper:

```bash
ur5e-infer-init
```

After arranging the scene, run:

```bash
ur5e-infer 20260905_150221:600 --execute
```

The defaults are the configuration verified smooth on the real robot: RTDE
servoJ with 500 Hz interpolation, `max-linear-speed=0.40`, 10 diffusion steps,
`lookahead=0.1`, `gain=300`, and continuous execution until `Ctrl+C`. The
robot-side loop is injected automatically and keeps publishing the final
setpoint during inference, avoiding socket-command-expiry braking.

The checkpoint may be a full path or `TRAIN_TIMESTAMP:EPOCH`. Add `--chunks N`
to limit execution or `--no-gripper` to isolate the arm.

## Socket comparison modes

The earlier task-completing baseline with smooth intra-chunk motion remains:

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition baseline \
  --smooth-alpha 0.7 --max-linear-speed 0.40 --chunks 0
```

The final-action-stretch experiment is also retained:

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition stretch \
  --smooth-alpha 0.7 --max-linear-speed 0.40 \
  --diffusion-steps 10 --chunks 0
```

Live testing showed clear deceleration both within and between chunks, so this
mode is not recommended. Stretching deliberately slows the final action, and
the next URScript interrupts that still-running `speedL`. It remains only for
comparison and does not affect the RTDE default.
