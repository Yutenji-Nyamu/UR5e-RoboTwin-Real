# Setup

[简体中文](../zh-CN/runbooks/setup.md)

1. Create or activate `RoboTwinSimReal` and install this repository editable.
2. Copy `configs/lab.example.yaml` to the ignored `configs/lab.yaml`.
3. Verify the B81L robot address, stable gripper by-id path, and two RealSense
   serial numbers.
4. Ensure the user is in `dialout` and `robotdata`; reconnect the USB serial
   device or log in again after group changes.
5. Verify `/data/robotics/ur5e-real` is mounted and writable.
6. Run `ur5e-real doctor --config configs/lab.yaml`.
7. With devices connected, run the same command with `--hardware`.

Automation should use `conda run -n RoboTwinSimReal ...` when shell activation
is not available. Site-specific Conda installation paths belong in local tooling,
not in this public repository.
