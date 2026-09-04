# Real-robot integration roadmap

[简体中文](zh-CN/ROADMAP.md)

Status: 2026-09-04. Devices, capture, and manual replay are commissioned; DP
offline data and servoJ are next. RoboTwin stays above one narrow policy adapter.

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
| Dual RealSense | Dual capture and 60-frame warmup tested | Live DP shadow input |
| socket replay | Session `20260903_182752` replayed completely | Keep as an independent regression path |
| RTDE servoJ | ACT client, recipe, and robot program exist | Hold, small step, ramp, watchdog |
| Diffusion Policy | Pinned upstream data, training, and six-action inference inspected | One-session Zarr and offline batch |

## Chunk execution

RoboTwin DP returns six actions. The first baseline executes all six in upstream
order before inferring again; it does not default to first-action-only execution
or overlapping-chunk fusion.

Before reaching the robot, every chunk passes:

1. finite values, coordinate meaning, and rotation-vector continuity;
2. workspace and per-step speed limits;
3. 10 Hz to 500 Hz interpolation for servoJ;
4. RTDE runtime state and measured TCP feedback.

Log raw prediction, commanded target, and measured TCP separately. Early
replanning or chunk fusion is compared only if the full six-step baseline shows
a measured problem. See [`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md).

## Motion backends

- `SocketMovelReplayBackend`: Remote mode, batched `movel`, open-loop timing;
  manual replay and optional comparison baseline only.
- `RtdeServoJBackend`: Local-mode robot program, 500 Hz setpoints and feedback;
  default learned-policy backend.

The backends never switch automatically. See
[`ROBOTWIN_INTEGRATION.md`](ROBOTWIN_INTEGRATION.md).

## Gates

### 0 — Offline contracts

- Add `/joint_action/vector`, a DP Zarr adapter, and round-trip tests.
- Test time alignment, rotation continuity, and interpolation with synthetic
  episodes.

Exit: one canonical episode produces a correct upstream DP batch and decodes to
the same single-arm meaning.

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

### 4 — servoJ backend

- Load the robot-side loop; test hold, millimetre step, slow ramp, update loss,
  tracking limit, and stop latency.

Exit: configured safety limits and measured rate/latency pass.

### 5 — Data and DP offline

- Capture one task episode; validate frames/timestamps/actions; convert to DP;
  overfit one episode; visualize all six predicted actions; benchmark latency.

Exit: repeatable checkpoint loading and sane chunks without robot output.

### 6 — Shadow then live

- Shadow: live inputs, predictions logged, no commands.
- Arm-only conservative chunk; compare socket and RTDE only if useful.
- Full six-step RTDE execution after trace review; enable debounced gripper last.

Exit: repeatable trials with stop reasons and raw/guarded/measured logs.

## Human–automation handoff

The human owns breakers, workcell clearance, emergency stop, Local/Remote mode,
URP loading/play, and safe start pose. Automation owns diagnostics, configs,
commands, logs, tests, plots, adapters, and commits. Immediately before any
output, the exact device, backend, and bounded action are stated once for human
confirmation.
