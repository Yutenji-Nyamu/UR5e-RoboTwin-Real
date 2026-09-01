# UR5e RoboTwin Real

[简体中文](README.zh-CN.md)

Reusable, one-stop infrastructure for B81L UR5e real-world data collection,
policy training, evaluation, and deployment—from device-level diagnostics to
RoboTwin integration.

```text
UR5e / gripper / RealSense
          ↓
hardware → control → collection/data → policy adapter
                                      ↓
                         pinned upstream RoboTwin
```

This repository owns real hardware, timing, data contracts, safety limits, and
deployment adapters. RoboTwin remains a reproducible pinned upstream framework
with only reviewed narrow patches; upstream policy code never directly owns a
socket, serial port, or camera.

## Start

```bash
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install -e .
cp configs/lab.example.yaml configs/lab.yaml
ur5e-real doctor --config configs/lab.yaml --hardware
```

For the current workstation, raw and converted data live under
`/data/robotics/ur5e-real` on the shared 4 TB disk.

## Main paths

- `examples/smoke/`: bounded one-device checks.
- `src/ur5e_real/hardware/`: RTDE, URScript, serial gripper, RealSense.
- `src/ur5e_real/control/`: trajectories and RTDE servoJ streaming.
- `src/ur5e_real/collection/` and `data/`: synchronized capture and conversion.
- `src/ur5e_real/adapters/`: narrow RoboTwin policy adapters.
- `robot_programs/`: PolyScope/RTDE robot-side programs.
- `integrations/robotwin/`: pinned-upstream metadata and reviewed patches.

ACT is the current real-robot baseline. Diffusion Policy is the next target; its
RoboTwin model configuration stays unchanged unless measured evidence justifies
a deviation. Manual socket replay remains an independent commissioning path.

## Documentation

See [`docs/README.md`](docs/README.md) for architecture, hardware commissioning,
storage/data management, migration records, and runbooks. Chinese mirrors are
maintained under [`docs/zh-CN`](docs/zh-CN).

## Repository policy

Local configuration, raw data, videos, checkpoints, caches, and the upstream
checkout are ignored. The historical repository remains unchanged. Commands
that can move hardware require an explicit execution step and a human at the
emergency stop.
