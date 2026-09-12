# Documentation

[简体中文](zh-CN/README.md)

## Design

- [Architecture](ARCHITECTURE.md)
- [RoboTwin integration boundary](ROBOTWIN_INTEGRATION.md)
- [RTDE read/write stack](RTDE_STACK.md)
- [Diffusion Policy real-robot implementation context](DIFFUSION_POLICY_PLAN.md)
- [Native RoboTwin π0.5 joint-space real-robot plan (current, Chinese)](plans/pi05/README.md)
- [π0.5 physical success record (Chinese)](plans/pi05/PHYSICAL_RESULT_20260910.md)
- [Cube RLT staged plan (current, Chinese; implementation available, not yet trained)](plans/pi05-rlt/CUBE_PLAN.md)
- [RLT commands, stage decisions and logs (Chinese)](plans/pi05-rlt/USAGE.md)
- [π0.5 and RLT initial research (historical, Chinese)](plans/pi05-rlt/README.md)
- [Real-robot roadmap](ROADMAP.md)
- [Migration record](MIGRATION.md)

## Data and workstation

- [Software, hardware, and layered debugging prerequisites](PREREQUISITES.md)
- [Storage layout](STORAGE.md)
- [Data management](DATA_MANAGEMENT.md)
- [Versioned experiment evidence and refresh command](experiments/README.md)

## Runbooks

- [New-machine commissioning and repair: start here](runbooks/new_machine.md)
- [Collection and replay quick reference](runbooks/operator_workflows.md)
- [Hardware commissioning](runbooks/hardware_commissioning.md)
- [Setup](runbooks/setup.md)
- [Collection](runbooks/collect.md)
- [Manual replay](runbooks/replay.md)
- [Diffusion Policy training](runbooks/train.md)
- [Diffusion Policy inference](runbooks/infer.md)

English documents are canonical for public interfaces. A maintained Chinese
mirror is available under [`zh-CN`](zh-CN); operational changes should update
both versions in the same commit.
