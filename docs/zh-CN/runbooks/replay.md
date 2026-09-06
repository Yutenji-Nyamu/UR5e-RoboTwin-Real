# 重播流程

[English](../../runbooks/replay.md)

这是独立的验机/物理回归链路。默认仍使用已经验证的socket批量 `movel`；新增的
RTDE分支可把同一条记录作为6步chunk，送入DP推理完全相同的servoJ执行层。

## 日常固定流程

开始前将PolyScope切到 **Remote Control**，确认夹爪没有夹持物体，清空原位路径及
整条录制轨迹覆盖的空间，人员守在急停旁。

第一条，初始化重播现场：

```bash
ur5e-replay-init
```

该命令内部自动选择环境和目录，检查硬件、低速回到固定原位并打开夹爪。

第二条，重播最新采集的轨迹，无需记时间戳：

```bash
ur5e-replay latest --execute
```

终端会先打印实际选择的run ID。重播指定历史轨迹时仍可传时间戳：

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

## RTDE chunk对照重播

现有socket重播没有变化；只有显式传入 `--backend rtde` 才进入新分支。先看摘要：

```bash
ur5e-replay latest --backend rtde
```

第一次只执行1个chunk：

```bash
ur5e-replay-init
ur5e-replay latest --backend rtde --chunks 1 --execute
```

确认后完整执行：

```bash
ur5e-replay latest --backend rtde --execute
```

与默认重播一样，正式进入记录动作前仍用低速socket `moveL`对齐第一帧；从第2帧目标
开始才进入共用RTDE执行层。

默认配置与当前DP推理执行层一致：记录的10 Hz TCP从第2帧开始作为动作，每6步一组，
组内由同一个 `stream_tcp_chunk` 插值为500 Hz servoJ设点，线速度上限 `0.40 m/s`、
`lookahead=0.1`、`gain=300`。每组之后保持最后设点 `0.08 s`，模拟当前10步DP推理的
典型空档；后台RTDE流不会中断。记录的按键事件会逐条原样重播，包括连续两次
`close`；只有旧数据没有事件表时，才用逐帧 `gripper_state` 经过推理共用的
`GripperPolicy` 兼容。

研究不同chunk边界时可显式覆盖：

```bash
ur5e-replay latest --backend rtde \
  --chunk-size 6 --chunk-gap 0.08 --max-linear-speed 0.40 --execute
```

例如 `--chunk-gap 0` 表示记录动作连续衔接、不模拟推理；`--chunk-gap 0.8` 可模拟较慢
推理。除上述起点对齐外，该分支不加载模型或相机；在动作执行边界下，它与在线推理
的区别就是“动作来自记录还是模型”。

## 边界

默认重播验证RTDE录制、socket运动、旋转向量连续性和夹爪事件。RTDE对照分支另外
验证与策略共用的servoJ执行层及chunk时序，但不加载或评价策略模型。
