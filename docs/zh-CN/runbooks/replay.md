# 重播流程

[English](../../runbooks/replay.md)

这是独立的验机/物理回归链路：机械臂使用socket批量 `movel`，夹爪按记录事件分段
执行。它不是策略训练或推理后端。

## 日常固定流程

开始前将PolyScope切到 **Remote Control**，确认夹爪没有夹持物体，清空原位路径及
整条录制轨迹覆盖的空间，人员守在急停旁。

第一条，初始化重播现场：

```bash
ur5e-replay-init
```

该命令内部自动选择环境和目录，检查硬件、低速回到固定原位并打开夹爪。

第二条，重播指定轨迹：

```bash
ur5e-replay 20260903_123456 --execute
```

参数是session的run ID；也可直接给 `session_*.json` 完整路径。命令会先低速对齐
轨迹起点，再完整执行记录的路径和夹爪事件。

## 第一次验证某条轨迹

先预览，不运动：

```bash
ur5e-replay 20260903_123456
```

然后初始化，并只执行第一个分段：

```bash
ur5e-replay-init
ur5e-replay 20260903_123456 --max-segments 1 --execute
```

确认没有位姿跳变、错误旋转分支、碰撞风险或意外夹爪事件后，该轨迹以后才使用上面
的日常完整重播命令。

## 边界

重播验证RTDE录制、socket运动、旋转向量连续性和夹爪事件。它是开环链路，不证明
策略模型、RTDE输入寄存器servoJ、chunk融合或闭环安全；学习策略保持独立servoJ
执行链路。
