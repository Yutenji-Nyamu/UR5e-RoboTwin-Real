# Replay workflow

[简体中文](../zh-CN/runbooks/replay.md)

This is an independent commissioning and physical-regression path: the arm uses
grouped socket `movel` programs and the gripper executes recorded events at
segment boundaries. It is not the policy training or inference backend.

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

Second, replay a named trajectory:

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
ur5e-replay 20260903_123456 --max-segments 1 --execute
```

Use the routine full replay only after confirming no pose jump, wrong rotation
branch, collision risk, or unexpected gripper event.

## Boundary

Replay validates RTDE recording, socket motion, rotation-vector continuity, and
gripper events. It is open loop and does not validate a policy model, RTDE
input-register servoJ, chunk blending, or closed-loop safety. Learned policies
keep the independent servoJ execution path.
