# Real-robot integration roadmap

[简体中文](zh-CN/ROADMAP.md)

Status: 2026-09-04. Devices, capture, replay, the offline DP path, and shadow
pass; a small servoJ test is next. RoboTwin stays above one narrow policy adapter.

## Fixed principles

1. Hardware, timestamps, safety, and execution belong to this repository.
2. RoboTwin owns model/training semantics. The pinned DP baseline (`8` horizon,
   `3` observations, `6` actions, `10 Hz`) is unchanged unless measurements and
   an explicit decision justify a deviation.
3. Manual record/replay and learned-policy execution are separate motion paths.
4. No first physical output is combined with another device test; any movement
   requires a human at the workcell.

## Boundary

```text
devices → canonical capture → dataset adapter → RoboTwin train/checkpoint
devices → live Observation → policy adapter → ActionChunk → guarded executor
```

The core contract is single-arm and physical:

- `Observation`: controller/monotonic timestamps, TCP pose `[6]`, gripper state,
  head/wrist frames, validity and calibration identity;
- `ActionChunk`: `N x 6` absolute TCP targets, `N` gripper targets, step duration,
  generation time and expiry.

Only a policy adapter may encode this into RoboTwin's dual-arm vector. For the
current 14-value compatibility layout the initial mapping is
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`.

## Current evidence

| Wheel | Status | Next physical gate |
|---|---|---|
| UR5e network/Dashboard | Reachable; PolyScope 5.13, `RUNNING`, safety `NORMAL` | Human verifies mode before output |
| RTDE receive | 10 Hz capture used for a complete session | Recheck 500 Hz feedback with servoJ |
| Serial gripper | Stable by-id path; open and close tested | Policy gripper decoding |
| Dual RealSense | Dual capture, 60-frame warmup, and DP shadow tested | Formal data capture |
| socket replay | Session `20260903_182752` replayed completely | Keep as an independent regression path |
| RTDE servoJ | ACT client, recipe, and robot program exist | Hold, small step, ramp, watchdog |
| Diffusion Policy | Zarr, native batch, GPU train, checkpoint load, offline, and shadow pass | One arm-only socket chunk, then gripper; servoJ separately |

## Chunk execution

RoboTwin DP returns six actions. The first baseline executes all six in upstream
order before inferring again; it does not default to first-action-only execution
or overlapping-chunk fusion.

Before reaching the robot, every chunk passes:

1. rotation-vector continuity and per-step speed limits;
2. 10 Hz to 500 Hz interpolation for servoJ;
3. RTDE runtime state and measured TCP feedback.

After measuring the full six-step baseline, decide whether more logging, early
replanning, or chunk fusion is useful. See
[`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md).

## Motion backends

- `SocketMovelReplayBackend`: Remote mode, batched `movel`, open-loop timing;
  manual replay only.
- `SocketSpeedLPolicyBackend`: RTDE state plus 10 Hz socket `speedl`, with
  selectable target EMA; first learned-policy commissioning backend.
- `RtdeServoJBackend`: robot program started automatically in Remote mode, with
  500 Hz RTDE setpoints and feedback; explicit experimental backend.

The backends never switch automatically. See
[`ROBOTWIN_INTEGRATION.md`](ROBOTWIN_INTEGRATION.md).

## Gates

### 0 — Offline contracts

- Add `/joint_action/vector`, a DP Zarr adapter, and round-trip tests.
- Test time alignment, rotation continuity, and interpolation with synthetic
  episodes.

Exit: one canonical episode produces a correct upstream DP batch and decodes to
the same single-arm meaning.

Status: complete.

### 1 — Read-only devices

- Dashboard/RTDE status, stable serial path, both camera streams, writable data
  root.

Exit: all independent checks pass. This gate is complete for the current setup.

### 2 — Bounded device outputs

- Gripper open/close with arm parked.
- Freedrive start/stop with the arm supported.

Exit: exact commands and stop behaviour are observed and recorded.

Status: complete.

### 3 — Manual record/replay

- Record a short path, inspect manifest/timing, dry-run replay, execute one
  segment, then compare recorded and replayed TCP/gripper traces.

Exit: no pose discontinuity and bounded tracking/timing error. This gate remains
independent of ML.

Status: one complete capture and replay is finished.

### 4 — policy motion backends

- Confirm socket speedL with one model chunk; separately test servoJ hold,
  millimetre step, slow ramp, tracking, and stop latency.

Exit: configured safety limits and measured rate/latency pass.

### 5 — Data and DP offline

- Capture one task episode; validate frames/timestamps/actions; convert to DP;
  overfit one episode; visualize all six predicted actions; benchmark latency.

Exit: repeatable checkpoint loading and sane chunks without robot output.

Status: conversion, training, reload, and shadow pipeline checks are complete;
the current checkpoint is not a task-quality model.

### 6 — Shadow then live

- Shadow: live inputs, predictions logged, no commands.
- Arm-only conservative chunk; compare socket and RTDE only if useful.
- Full six-step socket execution first; keep RTDE as an explicit A/B test;
  enable the debounced gripper last.

Exit: repeatable trials with stop reasons and raw/guarded/measured logs.

Status: shadow is complete; socket live execution is next, while servoJ remains
independently gated.

## Human–automation handoff

The human owns breakers, workcell clearance, emergency stop, Remote mode, and
the safe start pose. Automation owns control-program startup, diagnostics,
configs, commands, logs, tests, plots, adapters, and commits. Immediately before any
output, the exact device, backend, and bounded action are stated once for human
confirmation.
