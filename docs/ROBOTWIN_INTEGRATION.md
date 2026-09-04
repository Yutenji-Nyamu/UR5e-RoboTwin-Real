# RoboTwin integration boundary

[简体中文](zh-CN/ROBOTWIN_INTEGRATION.md)

## Location and ownership

`scripts/bootstrap_robotwin.sh` clones the repository named in `robotwin.lock`
into the ignored `.third_party/RoboTwin` directory and checks out commit
`210720340637cb4619283b295dde4cdd807c9e66`.

```text
UR5e_RoboTwin_Real (tracked, owned here)
├── hardware / control / collection / data
├── adapters/robotwin_act and adapters/robotwin_dp
├── integrations/robotwin (lock metadata and narrow patches)
└── .third_party/RoboTwin (ignored, reproducible upstream checkout)
```

Dependency direction is one-way: this repository's adapters may call RoboTwin;
RoboTwin does not import hardware modules, and core hardware never imports
RoboTwin. Training output, generated task config, and checkpoints remain runtime
artifacts and are not committed.

## Existing ACT real path

The migrated ACT implementation uses RTDE for both directions:

1. the workstation receives `actual_TCP_pose` and runtime state at 500 Hz;
2. ACT predicts a TCP/gripper action;
3. the workstation writes six TCP values to RTDE input registers;
4. the running PolyScope loop reads those registers, solves inverse kinematics,
   and calls `servoj` every 2 ms;
5. the serial gripper is commanded independently.

The model has `chunk_size=50`, but the current adapter applies only
`prediction[0,0]`; true chunk scheduling remains future work.

## Diffusion Policy target

The pinned RoboTwin DP baseline defines:

- horizon `8`;
- observation steps `3`;
- action steps `6`;
- runner frequency `10 Hz`;
- a 14-value dual-arm compatibility vector.

These upstream model semantics remain unchanged. The adapter executes all six
steps and applies only TCP velocity limits plus 10 Hz-to-500 Hz interpolation
below the model boundary.

The converter now produces upstream `/joint_action/vector` and a DP Zarr tested
with the native Dataset. The compatibility mapping is
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`, so collection, training,
and inference share exactly one meaning.
The first baseline keeps upstream head-only input and complete six-action
execution; see [`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md) for stages
and later decisions.

## Why not use socket replay as the DP default?

| Backend | Advantage | Limitation | Role |
|---|---|---|---|
| socket + batched `movel` | Very simple; already proven for recorded trajectories | Open-loop timing; replacing scripts can interrupt motion; no high-rate watchdog | Manual replay and optional first end-to-end baseline |
| RTDE input + servoJ | Continuous setpoints, measured state, tracking checks, watchdog, smooth interpolation | Robot-side loop still requires commissioning | Default ACT/DP learned-policy backend; program starts automatically |

A six-step DP chunk can be sent through either backend, so choosing RTDE does not
change the policy output. The ACT RTDE loop is now reused as DP's 500 Hz execution
layer. Socket remains the manual-replay backend; the two never switch automatically.
