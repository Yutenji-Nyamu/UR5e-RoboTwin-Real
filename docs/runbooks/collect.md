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

Raw data lives under `/data/robotics/ur5e-real/raw`. Each trajectory has one
`session_<run_id>.json` containing task, timing, outcome, code commit, counts,
and product paths. Later reviews do not rewrite image, RTDE, or gripper CSV
products.

List the latest 20 sessions:

```bash
ur5e-real sessions --config ~/UR5e_RoboTwin_Real/configs/lab.yaml
```

See [hardware commissioning](hardware_commissioning.md) for low-level checks.
