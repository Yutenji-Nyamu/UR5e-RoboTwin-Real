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

After arranging the scene, select one of three independent executors.

### A. Socket baseline (proven smooth within a chunk)

This reads RTDE feedback and sends `speedl` over socket 30001; no manually
started PolyScope RTDE program is required:

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition baseline \
  --smooth-alpha 0.7 --max-linear-speed 0.40 --chunks 0
```

This is the baseline that completed the task. `--smooth-alpha 0.7` is the
intra-chunk target EMA; `baseline` leaves the inference gap unchanged.

### B. Socket final-action stretch

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition stretch \
  --smooth-alpha 0.7 --max-linear-speed 0.40 \
  --diffusion-steps 10 --chunks 0
```

The first five actions are identical to the baseline. Only the final target is
spread over `100 ms + measured inference time + 20 ms`; the next chunk then
preempts it. There is no background thread or concurrent writer to port 30001.
The selected final `speedL` duration is printed for every chunk.

On this workstation, `--diffusion-steps 10` reduced inference from roughly 790
ms to 80 ms, with nearly unchanged mean error on three offline observations. It
changes inference sampling only and does not require retraining.

### C. RTDE servoJ at 500 Hz

The model, cameras, and observations remain unchanged; only execution differs:

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend rtde --max-linear-speed 0.40 --diffusion-steps 10 \
  --servoj-lookahead 0.1 --servoj-gain 300 --chunks 0
```

The command injects the robot-side servoJ loop and interpolates the six 10 Hz
targets into 500 Hz setpoints. It keeps publishing the final setpoint during
inference, so there is no socket-command-expiry brake, although the robot can
still pause briefly. `0.1/300` are the official servoJ defaults; explicit valid
ranges are `lookahead=0.03--0.2` and `gain=100--2000`. This backend is wired but
still awaits its first verified following test in this repository; change the
end to `--chunks 1 --no-gripper` for a one-chunk comparison.

The executors never switch automatically. `--chunks 0` runs until `Ctrl+C`; the
gripper consumes action element 14, and `--no-gripper` isolates arm motion.
