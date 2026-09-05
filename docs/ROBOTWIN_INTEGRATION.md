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

The first policy path confirmed to move the real arm used RTDE for
`actual_TCP_pose`, then sent bounded `speedl(..., t=0.1)` commands over socket
30001. Its later version applied a target EMA with alpha `0.7`. A separate ACT
RTDE-register/servoJ experiment also exists, but has not yet passed a known
displacement test in this repository.

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
steps, then explicitly selects either the historical socket speedL executor or
the experimental 500 Hz RTDE servoJ executor below the model boundary.

The converter now produces upstream `/joint_action/vector` and a DP Zarr tested
with the native Dataset. The compatibility mapping is
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`, so collection, training,
and inference share exactly one meaning.
The first baseline keeps upstream head-only input and complete six-action
execution; see [`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md) for stages
and later decisions.

## Motion backends

| Backend | Advantage | Limitation | Role |
|---|---|---|---|
| socket `speedl` | Based on the first successful ACT execution; no PolyScope RTDE program | 10 Hz command timing and a pause during slow inference | Default first DP commissioning path; target EMA is selectable |
| RTDE input + servoJ | 500 Hz setpoints and robot-side lookahead | Register-write motion still requires a known-displacement test | Explicit experimental backend |
| socket + batched `movel` | Proven for recorded trajectories | Open-loop segment timing | Manual replay only |

A six-step DP chunk can use either learned-policy backend without changing the
policy output. Select `--backend socket` or `--backend rtde`; the executor never
switches automatically.
