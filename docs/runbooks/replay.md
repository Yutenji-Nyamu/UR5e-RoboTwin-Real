# Manual record/replay

This is a commissioning and regression tool, not a required stage of policy
training or inference. It replays a demonstrated TCP path with two independent
outputs:

- arm: grouped `movel` URScript sent to port `30001`;
- gripper: serial open/close events aligned to segment boundaries.

Recording reads `actual_TCP_pose` from RTDE and can enable freedrive. Replay is
open-loop: completion is estimated from planned segment duration, so it is not
the backend for learned-policy rollout.

## 1. Dry-run

Replay is dry-run by default. First inspect point filtering, gripper-event
segmentation, and estimated durations:

```bash
ur5e-real replay --config configs/lab.yaml ACTION.csv --gripper-events EVENTS.csv
```

No robot connection is made without `--execute`.

## 2. First physical segment

The human operator must:

1. clear the workcell and hold the emergency-stop position;
2. verify the start pose is close to the recording start pose;
3. switch PolyScope to Remote Control and clear safety faults;
4. confirm the gripper and tool cannot collide during the first segment.

Then run only one segment:

```bash
ur5e-real replay --config configs/lab.yaml ACTION.csv \
  --gripper-events EVENTS.csv --max-segments 1 --execute
```

Stop immediately for a pose jump, wrong rotation branch, unexpected gripper
event, or timing mismatch. Expand to the full trajectory only after inspecting
the one-segment result.

## 3. What this validates

Passing replay establishes that RTDE capture, rotation-vector continuity,
socket motion, serial gripper commands, and the physical calibration agree. It
does **not** validate the policy model, RTDE input-register servo loop, chunk
blending, or learned closed-loop safety; those have separate gates in
[`../ROADMAP.md`](../ROADMAP.md).
