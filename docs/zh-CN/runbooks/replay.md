# 手动录制轨迹重播

[English](../../runbooks/replay.md)

这是独立的验机和物理回归链路，不是策略训练/推理后端。机械臂使用 socket 批量
`movel`，夹爪按录制事件分段执行；RTDE在这里负责录制TCP位姿。

## 1. 无运动预检

直接传入采集生成的 session JSON；程序会自动找到动作和夹爪CSV：

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
ur5e-real replay --config configs/lab.yaml \
  /data/robotics/ur5e-real/raw/action/session_RUN_ID.json
```

默认只打印点数、分段、夹爪事件和预计时长，不连接机器人。

## 2. 只执行第一个物理分段

现场人员清空工作区并守在急停旁，确认当前TCP接近录制起点，PolyScope为 Remote
Control 且无安全故障，然后：

```bash
ur5e-real replay --config configs/lab.yaml \
  /data/robotics/ur5e-real/raw/action/session_RUN_ID.json \
  --max-segments 1 --execute
```

确认没有位姿跳变、错误旋转分支、碰撞风险或意外夹爪事件后，去掉
`--max-segments 1` 才是完整重播。`--execute` 会先以低速移动到记录起点，因此每次
都必须现场确认。

旧数据没有session JSON时仍可直接提供动作CSV，并用 `--gripper-events` 指定事件CSV。

## 3. 边界

这条链路验证RTDE录制、socket运动、旋转向量连续性、夹爪事件和物理工作区是否
一致。它是开环重播，不证明策略模型、RTDE输入寄存器servoJ、chunk融合或闭环安全；
学习策略继续使用独立servoJ链路。
