# Real-robot integration roadmap

Status snapshot: 2026-09-01. This is the execution order for rebuilding the
stack from device wheels upward while keeping RoboTwin above a narrow adapter.

## Boundary and target flow

```text
our repository                                  upstream RoboTwin
--------------                                  ----------------
UR5e / gripper / cameras
        |
capture + canonical episode  ---- adapter ----> DP data / train / checkpoint
        |
live observation              ---- adapter ----> DP inference
        |                                            |
        +---- safety-aware ActionChunk executor <----+
                     |
              RTDE servoJ + gripper
```

RoboTwin owns policy implementations and their training conventions. This
repository owns every physical device, timestamps, calibration, canonical data,
adapter mapping, motion limits, watchdogs, and execution. Upstream source stays
pinned under the ignored `.third_party/RoboTwin`; local changes enter only as
small reviewed patches or adapters.

## What is already known

| Area | Current evidence | Meaning |
|---|---|---|
| RTDE receive | Historical collection records `actual_TCP_pose` at 10 Hz | This is the proven state source |
| Socket write | Historical and migrated replay sends batched `movel` URScript | Keep for manual replay and commissioning |
| Replay smoothing | Rotation-vector branch fixing, waypoint filtering, event segmentation | Reusable pure math, but replay remains open-loop |
| RTDE servoJ | A 500 Hz input-register client and robot-side program exist | Intended learned-policy backend; must be commissioned independently |
| ACT | Real adapter loads a chunked model but currently applies only `prediction[0,0]` | Useful baseline, not yet a true chunk executor |
| Diffusion Policy | Pinned upstream config uses horizon 8, 3 observation steps, 6 action steps at 10 Hz | One inference can supply about 0.6 s of future targets |

The old manual replay therefore is indeed “RTDE read + socket write”. It is the
latest exercised physical route, but it should not become the learned-policy
control loop.

## Canonical contracts

The core should not expose RoboTwin's simulated dual-arm naming.

`Observation`:

- monotonic and controller timestamps;
- measured TCP position + rotation vector (`6` floats);
- physical gripper state (`1` float plus discrete command state);
- head and wrist images with capture timestamps;
- validity flags and calibration/config identity.

`ActionChunk`:

- `N x 6` absolute TCP targets in one declared frame;
- `N` gripper targets;
- policy step duration `dt`, generation timestamp, and expiry;
- optional uncertainty/debug metadata.

The RoboTwin DP adapter performs the compatibility mapping. For the pinned
14-dimensional policy, the initial explicit convention is
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`. The upstream dataset calls
this `joint_action/vector`, but our values remain TCP poses. This is acceptable
only when training and inference use exactly the same tested mapping.

Immediate dataset gap: the current converter creates the four split datasets
but not `/joint_action/vector`, while upstream DP `process_data.py` requires it
and treats sample `t+1` as the action for sample `t`. The DP adapter must add and
round-trip-test this vector before training.

## Chunk execution and smoothing

Diffusion Policy returning six actions is useful, but the first real rollout
must not blindly execute the whole 0.6-second chunk open-loop. Use receding
horizon control:

1. obtain the newest measured observation and infer a six-step chunk;
2. reject stale/non-finite/out-of-frame targets;
3. make rotation vectors continuous with the measured pose;
4. clamp workspace, translation/rotation velocity, acceleration, and jerk;
5. align the chunk head to the current measured pose and blend overlap with the
   still-valid tail of the previous chunk;
6. interpolate 10 Hz policy targets to the 500 Hz servoJ stream with a
   minimum-jerk trajectory;
7. execute only `K=1` policy step initially, then replan; increase to `K=2` only
   after latency and tracking-error measurements are safe;
8. stop on stale chunks, RTDE loss, PolyScope runtime stop, workspace violation,
   or sustained tracking error.

Log three trajectories separately: raw model output, safety/blended target, and
measured execution. That makes smoothing auditable instead of hiding errors.

## Two deliberately separate motion paths

| Path | PolyScope/control mode | Purpose | Feedback |
|---|---|---|---|
| `SocketMovelReplayBackend` | Remote Control; port 30001 | Human-recorded trajectory replay and regression | RTDE is recorded, but segment execution is time-estimated/open-loop |
| `RtdeServoJBackend` | Local mode with the robot-side servoJ program running | ACT/DP shadow-to-live rollout | 500 Hz measured pose, input registers, watchdog |

Do not automatically fall back from servoJ policy control to socket `movel`.
Changing backend also changes safety assumptions and requires an explicit lab
step.

## Commissioning gates

### Gate 0 — offline contracts

- Unit-test pose/rotation continuity, time alignment, bounds, chunk overlap, and
  10-to-500 Hz interpolation.
- Add `/joint_action/vector` and a DP Zarr adapter; verify dimensions, camera
  order, one-step shift, and normalization on a synthetic episode.
- Pin dependencies and make all commands dry-run without devices.

Exit: tests prove a canonical episode can make a DP batch and decode back to
the same single-arm semantics.

### Gate 1 — devices without motion

- RTDE connect/read and timestamp-rate report.
- Gripper serial identify/open/close with the arm safely parked.
- Head and wrist camera serial, frame rate, exposure, and timestamp report.
- Confirm the 4 TB data root is mounted and writable before recording.

Exit: each device has an independent smoke command and saved diagnostic report.

### Gate 2 — manual physical replay

- In Remote Control, test freedrive start/stop and capture a short trajectory.
- Dry-run replay, then execute one segment with a human at the emergency stop.
- Compare recorded and replayed TCP traces and gripper event timing.

Exit: bounded one-segment tracking and no discontinuity; then allow a full manual
replay. This gate is independent of ML.

### Gate 3 — servoJ backend

- In Local mode, load/run the robot-side program.
- Test pose hold, a millimetre-scale step, then a slow minimum-jerk ramp.
- Deliberately stop updates and verify watchdog/stop behaviour.

Exit: measured rate, latency, tracking error, stop latency, and workspace clamps
meet configured limits.

### Gate 4 — collection quality

- Record one task episode; produce a manifest and timing/alignment report.
- Preview the converted HDF5 and verify TCP/gripper/images at several events.
- Use the independent replay path as a physical equivalence check.

Exit: no missing/duplicate frames beyond declared tolerance and all coordinate
frames/actions are documented.

### Gate 5 — DP offline

- Convert episodes to the pinned DP Zarr layout.
- Overfit one short episode, run offline forward inference, and visualize all
  six predicted targets against ground truth.
- Benchmark end-to-end inference latency on this GPU.

Exit: repeatable training/checkpoint loading and numerically sane chunks without
connecting the robot.

### Gate 6 — shadow then live

- Shadow mode: robot stationary, live sensors in, chunks logged but not sent.
- Live arm only: gripper disabled, `K=1`, conservative limits.
- Then `K=2`, enable overlap blending, finally enable debounced gripper actions.
- Run the complete task only after each earlier trace passes review.

Exit: reproducible task trials with stop reasons, timing, raw/blended/executed
logs, and no unbounded motion command.

## Human–Codex handoff

The operator handles physical facts Codex cannot safely infer: power breakers,
workcell clearance, emergency stop, PolyScope Local/Remote mode, loading and
starting a URP, and placing the robot at a safe start pose. Codex handles config,
diagnostics, commands, logs, tests, adapters, plots, and commits. Immediately
before any movement, Codex states the exact backend and bounded motion; the
operator confirms the physical setup once.

The next practical session starts at Gate 1: power the robot, connect the head
camera, keep the arm parked, and run the no-motion diagnostics. Only after those
pass should we ask PolyScope to enter either replay mode or servoJ mode.
