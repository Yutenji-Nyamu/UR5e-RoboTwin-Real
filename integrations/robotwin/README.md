# RoboTwin integration boundary

[简体中文](README.zh-CN.md)

`RoboTwin` is third-party upstream code. It is pinned to the exact historical
version in `robotwin.lock` and cloned into `.third_party/RoboTwin` by
`scripts/bootstrap_robotwin.sh`.

Our current code is kept in `src/ur5e_real/adapters/robotwin_act`; the planned
DP integration will live beside it in `adapters/robotwin_dp`. The ACT adapter adds:

- real HDF5 to ACT preprocessing;
- real dual-camera/TCP/gripper inference;
- the task configuration written during preprocessing;
- one narrow, documented compatibility patch.

Do not commit `.third_party/RoboTwin`, trained checkpoints, or generated
`SIM_TASK_CONFIGS.json`. If upstream is upgraded, update the lock and revalidate
the patch and the complete hardware runbook separately.

See `docs/ROBOTWIN_INTEGRATION.md` for the ownership map, unchanged DP baseline,
and socket-versus-RTDE execution decision.
