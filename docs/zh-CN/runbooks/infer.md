# RoboTwin DP 推理

[English](../../runbooks/infer.md)

底层命令可先进入统一环境和仓库：

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
```

## 1. 离线推理

```bash
ur5e-real infer-dp --checkpoint CHECKPOINT \
  --episode HDF5_EPISODE --index 20 --output prediction.npz
```

读取录制图像和状态，输出6步预测、耗时及与标签的误差；不连接硬件。

## 2. 真机 shadow

```bash
ur5e-infer 20260905_150221:600 --shadow --chunks 10
```

连接双相机和只读RTDE，只打印预测，不发送机械臂或夹爪命令。`--chunks 0` 表示运行
到 `Ctrl+C`。checkpoint可传完整路径，或传 `训练时间戳:epoch`，避免多个同名
`600.ckpt` 混淆。

## 3. 真机执行

PolyScope保持 **Remote Control**。初始化会自动检查设备、低速回原位并打开夹爪：

```bash
ur5e-infer-init
```

布置场景后直接运行：

```bash
ur5e-infer 20260905_150221:600 --execute
```

默认即为实机验证理想的 RTDE servoJ 配置：500 Hz插值、`max-linear-speed=0.40`、
`diffusion-steps=10`、`lookahead=0.1`、`gain=300`，持续运行到 `Ctrl+C`。程序自动发送
机器人端servoJ循环；推理期间持续保持最后设点，不发生socket命令超时式硬刹。

checkpoint可传完整路径或 `训练时间戳:epoch`。需要限制长度时追加 `--chunks N`；
只看机械臂时追加 `--no-gripper`。

## socket对照版本

此前成功完成任务、chunk内平滑的基线仍保留：

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition baseline \
  --smooth-alpha 0.7 --max-linear-speed 0.40 --chunks 0
```

末步拉伸实验也保留：

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition stretch \
  --smooth-alpha 0.7 --max-linear-speed 0.40 \
  --diffusion-steps 10 --chunks 0
```

实测该实验在chunk内外均出现明显减速，不推荐继续使用。原因是末步主动降速，且下一
URScript会中断尚未结束的 `speedL`。它只作为对照保留，不影响RTDE默认配置。
