# UR5e RoboTwin Real

Real-robot support for a UR5e workcell, organized from hardware smoke tests to
data collection, conversion, replay, and RoboTwin ACT inference.

This repository contains **our real-world code only**. RoboTwin is an upstream
dependency pinned by [`robotwin.lock`](robotwin.lock) and checked out under the
ignored `.third_party/` directory. Real-robot adapters live under
`src/ur5e_real/adapters/`; they are never mixed into the upstream source tree.

## Architecture

```text
examples/smoke              one-device diagnostics
src/ur5e_real/hardware      URScript, RTDE, serial gripper, RealSense
src/ur5e_real/control       trajectories and RTDE servoJ streaming
src/ur5e_real/collection    synchronized arm/gripper/dual-camera collection
src/ur5e_real/data          HDF5 conversion and inspection
src/ur5e_real/replay.py     guarded trajectory replay
src/ur5e_real/adapters      RoboTwin policy-specific integration
integrations/robotwin       upstream bootstrap metadata and narrow patches
robot_programs              robot-side RTDE recipes/scripts
```

## Quick start

```bash
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install -e .
cp configs/lab.example.yaml configs/lab.yaml
# Edit configs/lab.yaml for this workcell.
ur5e-real doctor --config configs/lab.yaml
```

Use an existing environment without recreating it:

```bash
conda run -n RoboTwinSimReal python -m pip install -e .
```

## Main workflow

```bash
# 1. Check dependencies; add --hardware only when devices are connected.
ur5e-real doctor --config configs/lab.yaml
ur5e-real doctor --config configs/lab.yaml --hardware

# 2. Collect synchronized RTDE + gripper + dual-camera data.
ur5e-real collect --config configs/lab.yaml

# 3. Convert raw runs to the RoboTwin-style episode schema.
ur5e-real convert --config configs/lab.yaml --task pick_block_bowl --task-config simple

# 4. Inspect replay without moving hardware; --execute is deliberately required.
ur5e-real replay --config configs/lab.yaml ACTION.csv --gripper-events EVENTS.csv
ur5e-real replay --config configs/lab.yaml ACTION.csv --gripper-events EVENTS.csv --execute

# 5. Checkout the pinned RoboTwin tree, then process/train/infer through our adapter.
scripts/bootstrap_robotwin.sh
```

Detailed procedures are in [`docs/runbooks`](docs/runbooks). Always keep a human
at the emergency stop for commands that can move the robot.

## Repository policy

- `configs/lab.yaml`, datasets, videos, HDF5 files, checkpoints, and upstream
  source are ignored.
- No API keys or unrelated RobotEnvironment/ReKep/CaP/RAG code are migrated.
- The historical project remains an archive and is not modified by this repo.
- Hardware-moving commands require an explicit `--execute` flag where practical.
