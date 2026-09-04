# Architecture

[简体中文](zh-CN/ARCHITECTURE.md)

Dependency direction is intentionally one-way:

```text
examples / CLI
      |
collection, data, replay, RoboTwin adapters
      |
control
      |
hardware

RoboTwin adapter ----> ignored pinned RoboTwin checkout
core hardware  -X->   RoboTwin
```

The hardware and data layers never import RoboTwin. This keeps device smoke
tests useful when ML dependencies or GPUs are unavailable. Policy adapters may
import upstream policy code only at runtime and only after the pinned checkout
has been bootstrapped.

The raw on-disk schema preserves the proven historical pipeline: RTDE and
gripper CSV files share a run ID with a dual-camera directory and a timestamp
alignment CSV. Conversion produces RoboTwin-style `episode*.hdf5` files. The
left arm and left wrist observations are placeholders because the physical
workcell currently has one arm and two cameras; the mapping is explicit in
`src/ur5e_real/data/schema.py`.

## Runtime boundary

The real-robot core uses physical meanings instead of RoboTwin's dual-arm field
names:

```text
RTDE + gripper + cameras
          |
          v
Observation(tcp_pose[6], gripper[1], images, timestamps)
          |
          v
RoboTwin policy adapter  <---->  pinned upstream checkout
          |
          v
ActionChunk(tcp targets[N,6], gripper[N], dt)
          |
          v
TCP velocity limits -> 10 Hz-to-500 Hz interpolation -> RTDE servoJ + serial gripper
```

Only an adapter may encode the single-arm seven-value state into an upstream
compatibility vector. For the current 14-value RoboTwin layout that mapping is
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`; the legacy HDF5 group name
`joint_action` does not change the fact that these six values are TCP pose, not
UR joint angles. Dataset round-trip tests now lock this convention.

The policy never owns sockets, serial ports, or cameras. It returns a chunk; the
real-robot executor handles velocity limits, interpolation, and transmission.
The detailed commissioning order is in
[`ROADMAP.md`](ROADMAP.md); upstream location and backend choices are in
[`ROBOTWIN_INTEGRATION.md`](ROBOTWIN_INTEGRATION.md).
