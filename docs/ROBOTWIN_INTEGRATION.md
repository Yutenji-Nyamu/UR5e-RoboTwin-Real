# RoboTwin integration boundary

[简体中文](zh-CN/ROBOTWIN_INTEGRATION.md)

## Location and ownership

`scripts/bootstrap_robotwin.sh` clones the repository named in `robotwin.lock`
into the ignored `.third_party/RoboTwin` directory and checks out commit
`210720340637cb4619283b295dde4cdd807c9e66`.

```text
UR5e_RoboTwin_Real (tracked, owned here)
├── hardware / control / collection / data
├── adapters/robotwin_act and future adapters/robotwin_dp
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

These are upstream model semantics and stay unchanged by default. Early
commissioning may cap how many predicted steps are physically sent, but that is
a safety gate—not a model/configuration change. Permanent changes to horizon,
frequency, action representation, or replanning require measured latency,
tracking, or task-success evidence and an explicit decision record.

The current converter still needs the upstream-required
`/joint_action/vector` and a tested DP Zarr adapter. The initial compatibility
mapping remains `[tcp(6), dummy_gripper, tcp(6), physical_gripper]` so collection,
training, and inference share exactly one meaning.
The first baseline keeps upstream head-only input and complete six-action
execution; see [`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md) for stages
and later decisions.

## Why not use socket replay as the DP default?

| Backend | Advantage | Limitation | Role |
|---|---|---|---|
| socket + batched `movel` | Very simple; already proven for recorded trajectories | Open-loop timing; replacing scripts can interrupt motion; no high-rate watchdog | Manual replay and optional first end-to-end baseline |
| RTDE input + servoJ URP | Continuous setpoints, measured state, tracking checks, watchdog, smooth interpolation | Requires a running PolyScope program and commissioning | Default ACT/DP learned-policy backend |

A six-step DP chunk can be sent through either backend, so choosing RTDE does not
change the policy output. Because the ACT RTDE loop already exists, socket motion
is not simpler overall for a closed-loop policy. We retain both behind an
explicit backend boundary and compare the same conservative test chunk; there is
never an automatic fallback between them.
