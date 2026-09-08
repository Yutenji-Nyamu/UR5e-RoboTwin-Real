# Collection workflow

[简体中文](../zh-CN/runbooks/collect.md)

## Routine workflow

Before starting, power the robot, gripper, and both cameras; switch PolyScope to
**Remote Control**; make sure the gripper holds no object; clear the home path
and workcell; and keep a person at the emergency stop.

First, initialize collection:

```bash
ur5e-collect-init
```

This command selects the `RoboTwinSimReal` environment and repository
configuration internally, checks all hardware, moves slowly to the shared home
pose, waits for RTDE arrival confirmation, and opens the gripper. Continue only
after `[READY]`.

Second, record one teleoperated trajectory:

```bash
ur5e-collect pick_block_bowl
```

`pick_block_bowl` is the trajectory task name. The command warms both cameras,
starts RTDE capture and freedrive, and then accepts:

- `c`: close gripper;
- `o`: open gripper;
- `q`: finish normally;
- `Ctrl+C`: finish as interrupted.

After stopping, enter `s`, `f`, or `a` to mark success, failure, or aborted;
press Enter to review later. Add a setup note when needed:

```bash
ur5e-collect pick_block_bowl --note "red block, trial 1"
```

## Output

Since the 2026-09-08 collector update, new sessions use **raw schema v3** and
record measured joints and TCP together. The command and existing keys are
unchanged. Before freedrive starts, the collector checks one complete RTDE
packet; missing/non-finite joint data is an error, never replaced with zeros.
The startup line identifies the new format:

```text
[STATE] schema=3 actual_q[6] + actual_qd[6] + TCP[6]; same RTDE packet
```

`rtde_tcp_gripper_<run_id>.csv` keeps its legacy name and first nine columns;
it appends `actual_q_0` through `actual_q_5` (rad), `actual_qd_0` through
`actual_qd_5` (rad/s), and `host_receive_time_s` (Unix seconds). These are actual
measurements, not IK estimates or desired joint targets. The manifest records
field meanings, joint order, units, and the pre-freedrive initial q/TCP/TCP-offset
snapshot. Host receive time is not camera exposure time or proof of hardware sync.

Keep recording at least one second after the final `o`, until physical release
is complete. On exit, `quality.last_open_to_last_frame_s` reports the recorded
image-tail span; a short/missing tail prints a warning. Quit and Ctrl+C are never
delayed to fill it. This timing check is not a success detector or a missing-frame
audit. Raw files are not trimmed; event-aware export trimming remains later work.

Raw data lives under `/data/robotics/ur5e-real/raw`. Each trajectory has one
`session_<run_id>.json` containing task, timing, outcome, code commit, counts,
and product paths. Later reviews do not rewrite image, RTDE, or gripper CSV
products.

The session list now shows `SCHEMA` and `STATE`: old v1/v2 records are `tcp`,
new populated dual-state records are `joint+tcp`, and zero-sample runs are `none`.
Existing DP/ACT conversion and replay still read TCP, including from the new CSV;
joint-space training/execution is not enabled by this collector update.

List the latest 20 sessions:

```bash
ur5e-real sessions --config ~/UR5e_RoboTwin_Real/configs/lab.yaml
```

See [hardware commissioning](hardware_commissioning.md) for low-level checks.
