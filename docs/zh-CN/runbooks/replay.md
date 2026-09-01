# 手动录制/重播

[English](../../runbooks/replay.md)

这是验机和回归工具，不是策略训练或推理的必经步骤。它通过两个独立输出重播
示教TCP路径：

- 机械臂：向端口 `30001` 发送成组 `movel` URScript；
- 夹爪：在分段边界对齐串口开/闭事件。

录制时从RTDE读取 `actual_TCP_pose`，并可启用freedrive。重播是开环的，完成时间
按规划段时长估计，因此不是学习策略默认后端。

## 1. Dry-run

先检查点过滤、夹爪事件分段和预计时长：

```bash
ur5e-real replay --config configs/lab.yaml ACTION.csv --gripper-events EVENTS.csv
```

没有 `--execute` 时不会连接机器人。

## 2. 首个物理分段

现场人员必须：

1. 清空工作区并守在急停位置；
2. 确认当前起点接近录制起点；
3. 将PolyScope切到Remote Control并清除安全故障；
4. 确认首段中夹爪和工具不会碰撞。

然后只执行一段：

```bash
ur5e-real replay --config configs/lab.yaml ACTION.csv \
  --gripper-events EVENTS.csv --max-segments 1 --execute
```

若出现位姿跳变、错误旋转分支、意外夹爪事件或时序不符，立即停止。检查一段结果
后才能扩展到完整轨迹。

## 3. 它证明什么

重播通过可证明RTDE采集、旋转向量连续性、socket运动、串口夹爪和物理标定一致；
它不能证明策略模型、RTDE输入寄存器servoJ循环、chunk融合或学习策略闭环安全。
后者有独立验收门，见 [`../ROADMAP.md`](../ROADMAP.md)。
