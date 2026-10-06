# Replay workflow

[简体中文](../zh-CN/runbooks/replay.md)

This is an independent commissioning and physical-regression path. The verified
RTDE chunk route is the default. It can
feed the same recording as six-action chunks into the exact servoJ execution
layer used by DP inference.

## Routine workflow

Switch PolyScope to **Remote Control**, make sure the gripper holds no object,
clear the home path and the complete recorded workspace, and keep a person at
the emergency stop.

First, initialize replay:

```bash
ur5e-replay-init
```

This command selects the environment and repository internally, checks the
hardware, moves slowly to the shared home pose, and opens the gripper.

Second, replay the most recently collected trajectory without remembering its
timestamp:

```bash
ur5e-replay latest --execute
```

The selected run ID is printed first. A historical trajectory can still be
selected explicitly:

```bash
ur5e-replay 20260903_123456 --execute
```

The argument is a session run ID; a full `session_*.json` path is also accepted.
The command first aligns slowly to the recorded start, then executes the complete
path and gripper events.

## First validation of a trajectory

Preview without motion:

```bash
ur5e-replay 20260903_123456
```

Then initialize and execute only the first segment:

```bash
ur5e-replay-init
ur5e-replay 20260903_123456 --chunks 1 --execute
```

Use the routine full replay only after confirming no pose jump, wrong rotation
branch, collision risk, or unexpected gripper event.

## RTDE chunk comparison replay

RTDE is selected when `--backend` is omitted. Preview its plan first:

```bash
ur5e-replay latest
```

Execute one chunk for the first live check:

```bash
ur5e-replay-init
ur5e-replay latest --chunks 1 --execute
```

Then run the complete recording:

```bash
ur5e-replay latest --inference-ms 250 --execute
```

A low-speed socket `moveL` aligns the first recorded frame
before trajectory execution. The shared RTDE executor begins with the second
frame target.

Defaults match the current DP executor. Recorded 10 Hz TCP rows beginning at
the second frame become actions, grouped six at a time. The same
`stream_tcp_chunk` interpolates each group into a 500 Hz servoJ stream with a
`0.40 m/s` linear limit, `lookahead=0.1`, and `gain=300`. The controller holds
the final setpoint for `0.08 s` between groups to imitate the typical current
ten-step DP inference gap; its background RTDE stream remains active. Recorded
button events are replayed exactly, including repeated `close` commands. Legacy
data without an event file falls back to per-frame `gripper_state` through the
inference `GripperPolicy`.

Override the chunk boundary explicitly for experiments:

```bash
ur5e-replay latest --inference-ms 250 --execute
```

`--inference-ms` specifies the hold **between chunks**, not per action: default
80 ms, 250 for a 250 ms model, or 0 for continuous replay. There is no extra hold
before the first or after the last chunk. The background RTDE stream keeps the
last setpoint throughout the hold. This simulates elapsed time, not model output.
The old seconds option `--chunk-gap 0.25` is equivalent to `--inference-ms 250`;
the two cannot be combined. Values must be finite and nonnegative.

Use `--backend socket` for the previous grouped `movel` path; `--max-segments`
belongs to that backend, while `--chunks` limits RTDE replay. Apart from the start alignment,
this branch loads neither model nor cameras. Below the action boundary, its
difference from online inference is whether actions come from the recording or
the model.

## Boundary

Default replay validates recorded TCP actions, the shared RTDE servoJ executor,
rotation-vector continuity, chunk timing, and gripper events. It does not load
or evaluate a policy model. The new options were checked with dry runs and mock
execution; no new physical replay was performed for this change.
