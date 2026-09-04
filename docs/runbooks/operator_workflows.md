# Collection and replay quick reference

[简体中文](../zh-CN/runbooks/operator_workflows.md)

These are the fixed daily entry points. They select the repository,
`RoboTwinSimReal` environment, and lab configuration automatically; no manual
`cd` or `conda activate` is required.

## Shared preparation

- Power the UR5e, gripper, and both cameras; connect camera and gripper cables.
- Switch PolyScope to **Remote Control**.
- Clear the home and working paths, and keep an operator at the emergency stop.

## Workflow 1: collection

1. Initialize:

   ```bash
   ur5e-collect-init
   ```

   This checks all devices, returns slowly to the fixed home pose, confirms
   arrival through RTDE, and opens the gripper. Continue after `[READY]`.

2. Start teleoperated collection:

   ```bash
   ur5e-collect TASK --note "optional setup note"
   ```

3. After `[READY] recording and freedrive are active`, move the arm manually.
   Use `c` to close, `o` to open, and `q` to finish; then enter `s`, `f`, or `a`
   for success, failure, or aborted.

The printed `[RUN] RUN_ID` is generated from the time automatically. Save it for
later use; raw products live under `/data/robotics/ur5e-real/raw`.

## Workflow 2: replay

1. Clear the home path and initialize:

   ```bash
   ur5e-replay-init
   ```

2. Restore the objects to their recorded starting positions and clear the whole
   trajectory workspace.

3. Execute the selected session:

   ```bash
   ur5e-replay RUN_ID --execute
   ```

The command aligns slowly to the recorded start, then executes the complete arm
path and gripper events. For an unvalidated trajectory, first omit `--execute`
to inspect its summary; see the [replay runbook](replay.md) for segmented testing.

## Boundary

Collection and manual replay form an independent commissioning loop. Replay
uses socket `movel`; future RoboTwin training and Diffusion Policy deployment
use the separate RTDE `servoJ` policy-execution path.
