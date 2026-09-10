# Real-robot integration roadmap

[简体中文](zh-CN/ROADMAP.md)

Current milestone (2026-09-10): native π0.5 joint-space cube execution succeeded;
see the [physical record](plans/pi05/PHYSICAL_RESULT_20260910.md),
[cube RLT next-stage plan](plans/pi05-rlt/CUBE_PLAN.md), and
[new-machine checklist](runbooks/new_machine.md). RLT is planning-only.
The remainder is the dated TCP/DP roadmap; its six-point chunks do not specify π0.5's K20.

Status: 2026-09-06. Devices, capture, replay, DP training/shadow, and live RTDE
servoJ inference pass; servoJ is now the default executor. RoboTwin stays above
one narrow policy adapter.

## Fixed principles

1. Hardware, timestamps, safety, and execution belong to this repository.
2. RoboTwin owns model/training semantics. The pinned DP baseline (`8` horizon,
   `3` observations, `6` actions, `10 Hz`) is unchanged unless measurements and
   an explicit decision justify a deviation.
3. Socket replay remains an independent regression path; an explicit RTDE
   replay mode may enter the shared action executor for controlled comparison.
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
| RTDE chunk replay | Recorded-action parsing and real-session dry-run pass | Execute one bounded chunk, then the complete session |
| RTDE servoJ | Smooth live execution of complete 500 Hz DP chunks | Quantify tracking error when useful |
| Diffusion Policy | Conversion, training, offline, shadow, and live task execution pass | Add consistent data and evaluate success rate |

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
  the default manual replay.
- `SocketSpeedLPolicyBackend`: RTDE state plus 10 Hz socket `speedl`, retained
  for comparison.
- `RtdeServoJBackend`: robot program started automatically in Remote mode, with
  500 Hz RTDE setpoints and feedback; the default policy executor and explicit
  recorded-action comparison backend.

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

Status: one complete capture/socket replay is finished. RTDE recorded-action
replay passes parsing and dry-run; its first physical chunk remains to test.

### 4 — policy motion backends

- Compare socket speedL and servoJ tracking and stop behavior.

Exit: configured safety limits and measured rate/latency pass.

### 5 — Data and DP offline

- Capture one task episode; validate frames/timestamps/actions; convert to DP;
  overfit one episode; visualize all six predicted actions; benchmark latency.

Exit: repeatable checkpoint loading and sane chunks without robot output.

Status: conversion, training, reload, and shadow pipeline checks are complete;
the current checkpoint is not a task-quality model.

### 6 — Shadow then live

- Shadow: live inputs, predictions logged, no commands.
- Execute complete six-action chunks and gripper output through RTDE servoJ.
- Select socket explicitly only when a comparison is useful.

Exit: repeatable trials with stop reasons and raw/guarded/measured logs.

Status: shadow and a complete live RTDE servoJ task pass; the next focus is data
volume and task-success evaluation.

## Human–automation handoff

The human owns breakers, workcell clearance, emergency stop, Remote mode, and
the safe start pose. Automation owns control-program startup, diagnostics,
configs, commands, logs, tests, plots, adapters, and commits. Immediately before any
output, the exact device, backend, and bounded action are stated once for human
confirmation.
