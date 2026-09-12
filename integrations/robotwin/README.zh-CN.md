# RoboTwin 集成边界

[English](README.md)

`RoboTwin` 是第三方上游代码。版本固定在 `robotwin.lock`，由
`scripts/bootstrap_robotwin.sh` 克隆到 `.third_party/RoboTwin`。

当前适配代码位于 `src/ur5e_real/adapters/robotwin_act` 和
`src/ur5e_real/adapters/robotwin_dp`，提供：

- 真实HDF5到ACT和DP格式的转换；
- 真实相机/TCP/夹爪观测与执行适配；
- RoboTwin原生DP训练与checkpoint加载；
- 预处理期间写入的任务配置；
- 一个窄且有记录的ACT兼容补丁。

独立的 `robotwin_pi05` 适配器新增UR关节数据、原生JAX训练/服务和专用joint servoJ执行器。
小型vendor补丁修复配置导入、锁定Orbax的asset回调和下载异常传播；模型和loss仍是原生实现。
当前进度和验证边界见 [π0.5实施记录](../../docs/plans/pi05/IMPLEMENTATION.md)。

不要提交 `.third_party/RoboTwin` 或模型权重。小型实验日志/配置和checkpoint元信息通过
[实验资料归档](../../docs/experiments/README.md)提交副本，不直接跟踪第三方目录下的生成文件。
升级上游时，应分别更新lock并重新验证补丁和完整真机
runbook。

归属关系、不修改的DP基线以及socket/RTDE执行选择见
`docs/ROBOTWIN_INTEGRATION.md`。
