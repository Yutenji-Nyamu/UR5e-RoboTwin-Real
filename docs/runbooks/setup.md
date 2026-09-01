# Setup

1. Create or activate `RoboTwinSimReal` and install this repository editable.
2. Copy `configs/lab.example.yaml` to the ignored `configs/lab.yaml`.
3. Fill the robot address, gripper device, and two RealSense serial numbers.
4. Ensure the user is in the `dialout` group and reconnect the USB serial device.
5. Run `ur5e-real doctor --config configs/lab.yaml`.
6. With devices connected, run the same command with `--hardware`.

Automation should use `conda run -n RoboTwinSimReal ...` when shell activation
is not available. Site-specific Conda installation paths belong in local tooling,
not in this public repository.
