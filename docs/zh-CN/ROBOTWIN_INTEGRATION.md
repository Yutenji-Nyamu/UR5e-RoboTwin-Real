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

第一次确认能够驱动真机的策略链路，是RTDE读取 `actual_TCP_pose`，再通过30001端口
发送限幅后的 `speedl(..., t=0.1)`；后续版本对目标做了 `alpha=0.7` 的EMA。旧项目也有
RTDE寄存器/servoJ实验，但本仓库尚未用已知位移证明它确实跟随。

模型设置了 `chunk_size=50`，但当前适配器只使用 `prediction[0,0]`；真正的 chunk
调度尚未实现。

## Diffusion Policy 目标

固定版本 RoboTwin 的 DP 基线为：

- horizon `8`；
- observation steps `3`；
- action steps `6`；
- 运行频率 `10 Hz`；
- 14维双臂兼容向量。

这些上游模型语义保持不改。适配层完整执行6步，下层明确选择历史socket speedL
执行器或实验性的500 Hz RTDE servoJ执行器。

当前转换器已生成 `/joint_action/vector` 和经过原生Dataset验证的DP Zarr。兼容映射为
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`，保证采集、训练、推理含义一致。
第一版沿用上游的头部单相机输入和完整6步执行；当前状态与待决策项见
[`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md)。

## 运动后端

| 后端 | 优点 | 局限 | 定位 |
|---|---|---|---|
| socket `speedl` | 基于第一次成功ACT执行；无需PolyScope RTDE程序 | 10 Hz命令；慢推理期间会暂停 | DP首次验机默认；目标EMA可调 |
| RTDE输入 + servoJ | 500 Hz设点和机器人端lookahead | 寄存器写运动仍需已知位移验证 | 明确的实验后端 |
| socket + 批量 `movel` | 已由录制轨迹重播验证 | 开环分段计时 | 仅手动重播 |

DP的6步chunk可通过任一策略后端执行，不改变模型输出。用 `--backend socket` 或
`--backend rtde` 明确选择；两者不自动切换。
