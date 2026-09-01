# Real-robot integration roadmap

[简体中文](zh-CN/ROADMAP.md)

Status: 2026-09-01. The stack is commissioned from individual devices upward;
RoboTwin stays above one narrow policy adapter.

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
| RTDE receive | Five stable 10 Hz TCP samples read | Longer rate/timestamp report |
| Serial gripper | Stable CH340 by-id device present | One open, then one close |
| Dual RealSense | Both configured D435i devices stream together | Timing/exposure report |
| socket replay | Historical and migrated batched `movel` path exists | Dry-run, then one segment |
| RTDE servoJ | ACT client, recipe, and robot program exist | Hold, small step, ramp, watchdog |
| Diffusion Policy | Pinned upstream code/config inspected | Dataset adapter and offline batch |

## Chunk execution

RoboTwin DP returns six actions. The target implementation preserves that output
and can execute all six after commissioning. For the first bounded motion only,
the executor may cap physical output to the first step; this is a temporary
safety gate, not a changed DP default or a permanent receding-horizon decision.

Before reaching the robot, every chunk passes:

1. finite/frame/staleness validation;
2. rotation-vector continuity;
3. workspace, velocity, acceleration, and jerk limits;
4. overlap blending where configured;
5. 10 Hz to 500 Hz minimum-jerk interpolation for servoJ;
6. RTDE loss, runtime stop, expiry, and tracking-error watchdogs.

Log raw prediction, guarded target, and measured execution separately. Any
proposal to execute fewer/more actions or replan earlier is evaluated from these
logs against the unchanged six-step baseline.

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
- Test time alignment, rotation continuity, limits, chunk overlap, and
  interpolation with synthetic episodes.

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

### 3 — Manual record/replay

- Record a short path, inspect manifest/timing, dry-run replay, execute one
  segment, then compare recorded and replayed TCP/gripper traces.

Exit: no pose discontinuity and bounded tracking/timing error. This gate remains
independent of ML.

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
