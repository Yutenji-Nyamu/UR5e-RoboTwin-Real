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
RoboTwin. The upstream worktree and model payloads stay outside Git. Small training logs,
configs and checkpoint metadata are versioned through the [evidence archive](experiments/README.md);
live runtime directories remain ignored.

## Existing ACT real path

The first policy path confirmed to move the real arm used RTDE for
`actual_TCP_pose`, then sent bounded `speedl(..., t=0.1)` commands over socket
30001. Its later version applied a target EMA with alpha `0.7`. A separate ACT
RTDE-register/servoJ experiment also exists in the old ACT path. The current DP
RTDE servoJ adapter has completed a smooth live task.

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
the verified 500 Hz RTDE servoJ executor below the model boundary.

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
| socket `speedl` | Based on the first successful ACT execution; no PolyScope RTDE program | Clear braking at chunk boundaries | Retained for comparison |
| RTDE input + servoJ | Smooth live motion with 500 Hz setpoints and robot-side lookahead | Holds the final setpoint during inference/gap | Default DP backend and explicit recorded-action comparison |
| socket + batched `movel` | Proven for recorded trajectories | Open-loop segment timing | Default manual replay |

A six-step DP chunk can use either learned-policy backend without changing the
policy output. Select `--backend socket` or `--backend rtde`; the executor never
switches automatically. `ur5e-replay --backend rtde` enters the same
`stream_tcp_chunk` executor below the policy boundary with recorded actions.
