# RoboTwin Diffusion Policy 真机实施上下文

[English](../DIFFUSION_POLICY_PLAN.md)

状态：2026-09-04。数据转换、官方训练、离线推理和真机shadow已通过；servoJ实机
执行是下一阶段。具体命令见训练与推理runbook。

## 目标与边界

```text
真机 raw session
  -> 规范 HDF5
  -> RoboTwin DP Zarr
  -> 训练与 checkpoint
  -> 离线回放评估
  -> 在线 shadow inference
  -> RTDE servoJ 实机执行
```

本仓库负责采集、数据语义、真机观测和执行；固定版本 RoboTwin 只负责 DP 模型、
训练和 checkpoint 语义。上游工作树位于 `.third_party/RoboTwin`，固定提交为
`210720340637cb4619283b295dde4cdd807c9e66`。

## 已确认的基础

| 部分 | 当前结论 |
|---|---|
| 采集 | UR5e RTDE、夹爪和双 D435i 的10 Hz同步采集已实机通过 |
| 数据 | session `20260903_182752` 已成功记录、复核和保存 |
| 重播 | 同一 session 的三段 socket `movel` 与两次夹爪事件已完整重播成功 |
| 旧 ACT | 已证明真实双相机观测、模型加载、RTDE读写和500 Hz servoJ的接法；旧实现每次只取ACT chunk第1步 |
| RoboTwin DP | HDF5 `joint_action/vector` -> Zarr -> Hydra训练 -> checkpoint -> 6步预测已实跑 |
| shadow | 双相机和只读RTDE已连续完成2个chunk；稳态推理约0.84秒，未发送命令 |
| servoJ | 客户端、RTDE recipe和机器人端脚本已迁入；新仓库内尚未重新做保持/小步/连续轨迹实测 |

一条数据足以验证转换、batch、前向、checkpoint加载和单episode过拟合；它不能用来
判断策略泛化或真实任务成功率。

## 第一版固定语义

| 项目 | 第一版决定 |
|---|---|
| 模型时间参数 | 沿用上游：horizon `8`、观测 `3` 步、输出 `6` 步、`10 Hz` |
| 图像 | 沿用上游基线，只训练 `head_camera -> head_cam`；腕部图像继续保存但暂不输入模型 |
| 真机状态 | 绝对 TCP `[x,y,z,rx,ry,rz]` 加物理夹爪状态，打开=`0`、闭合=`1` |
| 14维适配 | `[tcp(6), dummy_gripper=0, tcp(6), physical_gripper]` |
| 监督动作 | `action[t] = state[t+1]`，是绝对 TCP 目标，不是 delta |
| 旋转 | 写入训练数据前先连续化等价旋转向量，避免接近 pi 时的数值跳支 |
| chunk | 按上游顺序完整执行6步，再进行下一次推理 |
| 真机输出 | 6个10 Hz目标由底层插值为500 Hz RTDE servoJ设点；这不改变DP模型配置 |
| 夹爪 | 从每步第14维解码，沿用现有阈值/去抖逻辑 |

RoboTwin 仿真原始14维向量表示双臂关节位置；真机适配层复用它的形状，但明确编码为
单臂 TCP。真实数据训练和真实推理因此语义一致，但未经额外转换不能直接与仿真
关节数据混合。

第一版不做 chunk 重叠融合，也不提前重规划。以后只有推理延迟、跟踪误差或成功率
表明确有需要时，才单独比较这些策略。

## 实施阶段

### 1. 单条数据打通转换（完成）

- 在规范 HDF5 中增加 `/joint_action/vector`；
- 新增 `adapters/robotwin_dp`，生成上游所需的 Zarr：
  `data/head_camera`、`data/state`、`data/action`、`meta/episode_ends`；
- 保留源 run ID、schema版本和转换器提交；
- 用当前一条 session 检查图像、14维状态、next-step动作和episode边界。

当前轨迹生成227个transition；RoboTwin原生 `RobotImageDataset` 已读取
`(B,8,3,240,320)` 图像、`(B,8,14)` 状态和动作。

### 2. 单episode训练烟雾测试（完成）

- 补齐并锁定 DP Python 依赖；
- 一条数据时使用 `val_ratio=0`，因为无法同时划分训练集和验证集；
- 用较小 batch 和少量 epoch 验证loss下降、保存checkpoint并重新加载；
- 对同一轨迹画出6步预测与标签。

已在A6000上用batch 8完成2 epoch/每epoch 3步，并生成、加载两个checkpoint。这些
只是烟雾测试参数；正式基线仍使用 RoboTwin 原始配置。

### 3. 离线与在线 shadow（完成）

- 对录制数据逐帧推理，不连接机械臂输出，记录预测和推理耗时；
- 接入真实 RTDE 与头部相机，维持3步观测队列；
- 在线只保存原始6步预测，不发送设点。

离线预测与标签对比已保存；真机shadow已完成2个完整6步chunk。当前模型只是管线
验证模型，不用于判断任务效果。

### 4. 策略执行后端（进行中）

- 先用历史上已经运动成功的“RTDE读 + socket `speedl`写”验DP，目标EMA默认
  `alpha=0.7`且可调；
- RTDE servoJ保留为明确的A/B后端，先通过已知毫米位移再用于策略评估；
- 再接DP，先机械臂，最后启用夹爪。

正式推理保持Remote模式。`--backend socket`无需PolyScope RTDE程序；
`--backend rtde`会注入servoJ循环。两者不改变6步策略输出，也不自动切换。

### 5. 正式数据与评估

- 固定任务起点、成功标准和允许的场景变化；
- 先收集一小批一致示范，跑完整训练/评估，再决定是否扩充；
- checkpoint始终关联dataset ID、配置、seed、上游提交和本仓库提交。

## 已实现的代码入口

```text
src/ur5e_real/adapters/robotwin_dp/   # HDF5/Zarr、模型加载、观测与动作适配
src/ur5e_real/control/                # chunk定时与500 Hz执行
ur5e-real process-dp                  # HDF5 -> Zarr
ur5e-real train-dp                    # 调用固定上游训练代码
ur5e-real infer-dp                    # offline / shadow / execute
ur5e-infer-init; ur5e-infer           # 固定真机初始化与一键在线入口
```

先保持适配层很薄，不复制 RoboTwin 的模型代码，也不直接修改被忽略的上游工作树。

## 尚未阻塞当前工作的决策

当前离线实现不需要额外决策，默认使用现有 `pick_place_cube` 数据和上述基线。以下
问题到对应阶段再决定：

1. 批量采集前，明确任务成功标准、物体起点范围和计划采集数量；
2. 若要混合仿真数据，需决定改用真实关节角还是增加 TCP/关节的明确转换；
3. 只有头部基线跑通后，再决定是否把腕部相机加入模型；
4. 只有完整6步执行出现实测问题后，再讨论提前重规划或 chunk 融合。

最近的下一步是用带训练时间戳的checkpoint先执行一个仅机械臂socket chunk；通过后
启用夹爪，RTDE servoJ另行验证。
