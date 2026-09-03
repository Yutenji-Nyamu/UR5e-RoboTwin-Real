# Replay a manually recorded trajectory

[简体中文](../zh-CN/runbooks/replay.md)

This is an independent commissioning and physical-regression path, not the
policy training/inference backend. The arm receives grouped socket `movel`
programs and the gripper runs recorded events at segment boundaries; RTDE is the
TCP-pose recorder in this path.

## 1. No-motion preview

Pass the session JSON produced by capture; it resolves both CSV paths:

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
ur5e-real replay --config configs/lab.yaml \
  /data/robotics/ur5e-real/raw/action/session_RUN_ID.json
```

By default this only prints point, segment, gripper-event, and timing summaries.
It does not connect to the robot.

## 2. Execute only the first physical segment

Clear the workcell, hold the emergency-stop position, confirm that the current
TCP is near the recorded start, and put PolyScope in Remote Control with no
safety fault. Then run:

```bash
ur5e-real replay --config configs/lab.yaml \
  /data/robotics/ur5e-real/raw/action/session_RUN_ID.json \
  --max-segments 1 --execute
```

Only after confirming no pose jump, wrong rotation branch, collision risk, or
unexpected gripper event should `--max-segments 1` be removed for full replay.
`--execute` first moves slowly to the recorded start, so every run requires an
on-site check.

For legacy data without a session JSON, pass the action CSV directly and use
`--gripper-events` for its event CSV.

## 3. Boundary

This path validates RTDE recording, socket motion, rotation-vector continuity,
gripper events, and the physical workcell together. It is open-loop replay; it
does not validate a policy model, RTDE input-register servoJ, chunk blending, or
closed-loop safety. Learned policies keep the independent servoJ path.
