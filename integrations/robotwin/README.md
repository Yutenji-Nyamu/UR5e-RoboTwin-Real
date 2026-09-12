# RoboTwin integration boundary

[简体中文](README.zh-CN.md)

`RoboTwin` is third-party upstream code. It is pinned to the exact historical
version in `robotwin.lock` and cloned into `.third_party/RoboTwin` by
`scripts/bootstrap_robotwin.sh`.

Our adapters live in `src/ur5e_real/adapters/robotwin_act` and
`src/ur5e_real/adapters/robotwin_dp`. They add:

- real HDF5 conversion to ACT and DP formats;
- real camera/TCP/gripper observation and execution adapters;
- native RoboTwin DP training and checkpoint loading;
- the task configuration written during preprocessing;
- one narrow, documented ACT compatibility patch.

The independent `robotwin_pi05` adapter adds joint-space UR data, native JAX
training/serving, and a separate joint servoJ driver. Its small vendor patches fix
config import, the pinned Orbax asset callback, and download error propagation;
the model/loss remain native. See [pi05 implementation](../../docs/plans/pi05/IMPLEMENTATION.md).

Do not commit `.third_party/RoboTwin` or model weights. Version small experiment logs,
configs and checkpoint metadata through the [evidence archive](../../docs/experiments/README.md),
not by tracking generated files inside the upstream tree. If upstream is upgraded, update the lock and revalidate
the patch and the complete hardware runbook separately.

See `docs/ROBOTWIN_INTEGRATION.md` for the ownership map, unchanged DP baseline,
and socket-versus-RTDE execution decision.
