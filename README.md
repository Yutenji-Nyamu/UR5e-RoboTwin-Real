# UR5e RoboTwin Real

[简体中文](README.zh-CN.md)

Reusable infrastructure for B81L UR5e real-world data collection, training,
evaluation, and deployment—from device-level drivers to a narrow RoboTwin
integration layer. This repository owns real hardware and data handling;
RoboTwin remains a pinned upstream policy framework.

## Daily operation

After the one-time [command installation](docs/runbooks/setup.md), these commands
select the repository and `RoboTwinSimReal` environment automatically; no manual
`cd` or Conda activation is required.

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
ur5e-replay latest --execute
```

After initialization, restore the recorded scene before executing the replay.
See the [one-page operator guide](docs/runbooks/operator_workflows.md) for the
complete fixed procedure and safety checks.

Live DP inference:

```bash
ur5e-infer-init
ur5e-infer 20260905_150221:600 --execute
```

The timestamp and epoch identify a checkpoint. The default is the smooth
real-robot configuration: 500 Hz RTDE servoJ, 10 diffusion steps, and continuous
execution. Socket comparison modes remain available in
[DP inference](docs/runbooks/infer.md).

Live π0.5 inference (joint-space cube pick-and-place reported successful):

```bash
ur5e-pi05-infer-init
ur5e-pi05-infer 20260909_01:1000 --execute
```

Five demonstrations, 1000 SFT steps; H=50, execute K=20 at 10 Hz through
500 Hz joint servoJ. See [physical result](docs/plans/pi05/PHYSICAL_RESULT_20260910.md)
and [π0.5 usage](docs/plans/pi05/USAGE.md). These commands assume matching local
data, checkpoint and calibrated hardware; they do not download the experiment.

## Diffusion Policy quick path

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real

# Convert only raw sessions with outcome=success; save the printed HDF5_RUN
ur5e-real convert --config configs/lab.yaml \
  --task pick_place_cube --task-config simple

# N is the successful-episode count; save the printed ZARR_PATH
ur5e-real process-dp HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N \
  --output ZARR_PATH --trim-static-edges

# Full training: 600 epochs, checkpoints at epochs 300 and 600
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N

# Offline inference, then live shadow inference
ur5e-real infer-dp --checkpoint CHECKPOINT --episode HDF5_EPISODE --index 20
ur5e-infer 20260905_150221:600 --shadow --chunks 10
```

See [training](docs/runbooks/train.md) and [inference](docs/runbooks/infer.md)
for all options.

## Data and documentation

The [native π0.5 joint-space path](docs/plans/pi05/README.md) now covers dual
recording, SFT, checkpoint reload and physical cube execution. The existing TCP
DP path is unchanged. [Cube RLT](docs/plans/pi05-rlt/CUBE_PLAN.md) now has offline token/BC learners,
actor–critic updates and a four-episode human-labelled real-robot interface. Formal RLT training and physical
validation have not started. See [`ur5e-rlt` usage and logs](docs/plans/pi05-rlt/USAGE.md) (Chinese).

Data lives under `/data/robotics/ur5e-real` on the shared 4 TB disk. Architecture,
hardware commissioning, data management, training, and inference documentation
is indexed in [`docs/README.md`](docs/README.md).

For a clean-machine deployment or repair, start with the
[bottom-up commissioning checklist](docs/runbooks/new_machine.md). It connects
installation, individual device tests, data/model migration and end-to-end checks.

The Diffusion Policy path passes HDF5, Zarr, native training, checkpoint loading,
shadow inference, and smooth RTDE servoJ execution; see
[training](docs/runbooks/train.md) and [inference](docs/runbooks/infer.md).

Each raw trajectory has one `session_<RUN_ID>.json` index containing task,
timestamps, duration, counts, notes, review history, and
`success/failure/aborted`. Images, RTDE, and gripper files remain immutable;
failed or aborted sessions are excluded from training but never deleted
automatically. List them with
`ur5e-real sessions --config configs/lab.yaml`. HDF5 and Zarr retain source run
IDs and trim bounds, and every checkpoint configuration points back to its Zarr
dataset. Data changes are briefly logged in
`/data/robotics/ur5e-real/DATA_LOG.md`.

Commands that move hardware require PolyScope **Remote Control**, a clear
workspace, and an operator at the emergency stop.
