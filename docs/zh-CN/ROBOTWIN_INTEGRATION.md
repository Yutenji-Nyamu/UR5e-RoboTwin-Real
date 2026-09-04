# RoboTwin 集成边界

[English](../ROBOTWIN_INTEGRATION.md)

## 位置与归属

`scripts/bootstrap_robotwin.sh` 根据 `robotwin.lock`，把 RoboTwin 克隆到被 Git
忽略的 `.third_party/RoboTwin`，并固定在提交
`210720340637cb4619283b295dde4cdd807c9e66`。

```text
UR5e_RoboTwin_Real（本仓库跟踪、维护）
├── hardware / control / collection / data
├── adapters/robotwin_act 与 adapters/robotwin_dp
├── integrations/robotwin（版本信息与小补丁）
└── .third_party/RoboTwin（忽略、可复现的上游工作树）
```

依赖只能单向：本仓库适配层可以调用 RoboTwin；RoboTwin 不导入真机硬件模块，
底层硬件也不导入 RoboTwin。训练输出、生成的任务配置和 checkpoint 都是运行产物，
不提交到 Git。

## 现有 ACT 真机链路

迁移后的 ACT 读写都使用 RTDE：

1. 工作站以500 Hz接收 `actual_TCP_pose` 和运行状态；
2. ACT 输出 TCP/夹爪动作；
3. 工作站把6维 TCP 写入 RTDE 输入寄存器；
4. PolyScope 中运行的控制循环读取寄存器、求逆解，并每2 ms调用 `servoj`；
5. 串口夹爪独立执行。

模型设置了 `chunk_size=50`，但当前适配器只使用 `prediction[0,0]`；真正的 chunk
调度尚未实现。

## Diffusion Policy 目标

固定版本 RoboTwin 的 DP 基线为：

- horizon `8`；
- observation steps `3`；
- action steps `6`；
- 运行频率 `10 Hz`；
- 14维双臂兼容向量。

这些上游模型语义保持不改。适配层完整执行6步，只在下层做TCP速度限制与
10 Hz到500 Hz插值。

当前转换器已生成 `/joint_action/vector` 和经过原生Dataset验证的DP Zarr。兼容映射为
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`，保证采集、训练、推理含义一致。
第一版沿用上游的头部单相机输入和完整6步执行；当前状态与待决策项见
[`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md)。

## 为什么不默认用 socket 重播执行 DP？

| 后端 | 优点 | 局限 | 定位 |
|---|---|---|---|
| socket + 批量 `movel` | 简单；已在录制轨迹重播中验证 | 开环计时；替换脚本可能打断运动；缺少高频 watchdog | 手动重播，可作为首次端到端基线 |
| RTDE输入 + servoJ | 连续设点、状态反馈、误差检查、watchdog、平滑插值 | 机器人端循环仍需单独验机 | ACT/DP正式学习策略后端；程序自动启动 |

DP 的6步 chunk 可通过任一后端执行，因此选择 RTDE 不会改变模型输出。已有 ACT
RTDE 循环已复用为DP的500 Hz执行层。socket继续服务手动重播；两者不自动切换。
