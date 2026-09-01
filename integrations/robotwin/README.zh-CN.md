# RoboTwin 集成边界

[English](README.md)

`RoboTwin` 是第三方上游代码。版本固定在 `robotwin.lock`，由
`scripts/bootstrap_robotwin.sh` 克隆到 `.third_party/RoboTwin`。

当前代码位于 `src/ur5e_real/adapters/robotwin_act`；计划中的DP适配将并列放在
`adapters/robotwin_dp`。ACT适配提供：

- 真实HDF5到ACT的预处理；
- 真实双相机/TCP/夹爪推理；
- 预处理期间写入的任务配置；
- 一个窄且有记录的兼容补丁。

不要提交 `.third_party/RoboTwin`、训练checkpoint或生成的
`SIM_TASK_CONFIGS.json`。升级上游时，应分别更新lock并重新验证补丁和完整真机
runbook。

归属关系、不修改的DP基线以及socket/RTDE执行选择见
`docs/ROBOTWIN_INTEGRATION.md`。
