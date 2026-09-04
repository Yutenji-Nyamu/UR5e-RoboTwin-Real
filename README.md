# UR5e RoboTwin Real

[简体中文](README.zh-CN.md)

Reusable infrastructure for B81L UR5e real-world data collection, training,
evaluation, and deployment—from device-level drivers to a narrow RoboTwin
integration layer. This repository owns real hardware and data handling;
RoboTwin remains a pinned upstream policy framework.

## Daily operation

The installed commands select the repository and `RoboTwinSimReal` environment
automatically; no manual `cd` or Conda activation is required.

Collection:

```bash
ur5e-collect-init
ur5e-collect TASK --note "optional setup note"
```

Use `c` to close the gripper, `o` to open it, and `q` to finish; then enter
`s`, `f`, or `a` for success, failure, or aborted. Save the `[RUN]` ID printed by
the command—the timestamp is generated automatically.

Replay:

```bash
ur5e-replay-init
ur5e-replay RUN_ID --execute
```

After initialization, restore the recorded scene before executing the replay.
See the [one-page operator guide](docs/runbooks/operator_workflows.md) for the
complete fixed procedure and safety checks.

Live DP inference:

```bash
ur5e-infer-init
ur5e-infer 600 --execute
```

`600` is the checkpoint number; a full path is also accepted. Add `--chunks 1`
for the first commissioning run. See [DP inference](docs/runbooks/infer.md).

## Diffusion Policy quick path

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real

# Convert only raw sessions with outcome=success; save the printed HDF5_RUN
ur5e-real convert --config configs/lab.yaml \
  --task pick_place_cube --task-config simple

# N is the successful-episode count; save the printed ZARR_PATH
ur5e-real process-dp HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N

# Full training: 600 epochs, checkpoints at epochs 300 and 600
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N

# Offline inference, then live shadow inference
ur5e-real infer-dp --checkpoint CHECKPOINT --episode HDF5_EPISODE --index 20
ur5e-infer 600 --shadow --chunks 10
```

See [training](docs/runbooks/train.md) and [inference](docs/runbooks/infer.md)
for all options.

## Data and documentation

Data lives under `/data/robotics/ur5e-real` on the shared 4 TB disk. Architecture,
hardware commissioning, data management, training, and inference documentation
is indexed in [`docs/README.md`](docs/README.md).

The Diffusion Policy path now passes HDF5, Zarr, native training, offline
checkpoint loading, and live shadow inference. The RTDE servoJ execution entry
point is packaged and awaits its first small-motion commissioning run; see
[training](docs/runbooks/train.md) and [inference](docs/runbooks/infer.md).

Each raw trajectory has one `session_<RUN_ID>.json` index containing task,
timestamps, duration, counts, notes, review history, and
`success/failure/aborted`. Images, RTDE, and gripper files remain immutable;
failed or aborted sessions are excluded from training but never deleted
automatically. List them with
`ur5e-real sessions --config configs/lab.yaml`. HDF5 and Zarr retain source run
IDs, and every checkpoint configuration points back to its Zarr dataset.

Commands that move hardware require PolyScope **Remote Control**, a clear
workspace, and an operator at the emergency stop.
