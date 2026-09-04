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

## Data and documentation

Data lives under `/data/robotics/ur5e-real` on the shared 4 TB disk. Architecture,
hardware commissioning, data management, training, and inference documentation
is indexed in [`docs/README.md`](docs/README.md).

The Diffusion Policy path now passes HDF5, Zarr, native training, offline
checkpoint loading, and live shadow inference; see [training](docs/runbooks/train.md)
and [inference](docs/runbooks/infer.md).

Commands that move hardware require PolyScope **Remote Control**, a clear
workspace, and an operator at the emergency stop.
