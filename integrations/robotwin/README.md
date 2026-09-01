# RoboTwin integration boundary

`RoboTwin` is third-party upstream code. It is pinned to the exact historical
version in `robotwin.lock` and cloned into `.third_party/RoboTwin` by
`scripts/bootstrap_robotwin.sh`.

Our code is kept in `src/ur5e_real/adapters/robotwin_act`. The adapter adds:

- real HDF5 to ACT preprocessing;
- real dual-camera/TCP/gripper inference;
- the task configuration written during preprocessing;
- one narrow, documented compatibility patch.

Do not commit `.third_party/RoboTwin`, trained checkpoints, or generated
`SIM_TASK_CONFIGS.json`. If upstream is upgraded, update the lock and revalidate
the patch and the complete hardware runbook separately.
