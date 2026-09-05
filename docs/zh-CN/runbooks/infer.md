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
ur5e-infer 20260905_150221:300 --shadow --chunks 10
```

连接双相机和只读RTDE，只打印预测，不发送机械臂或夹爪命令。`--chunks 0` 表示运行
到 `Ctrl+C`。checkpoint可传完整路径，或传 `训练时间戳:epoch`，避免多个同名
`600.ckpt` 混淆。

## 3. 真机执行

PolyScope保持 **Remote Control**。初始化会自动检查设备、低速回原位并打开夹爪：

```bash
ur5e-infer-init
```

布置场景后，先测试基于旧ACT成功链路的socket版本；它用RTDE读状态、Socket 30001
发送 `speedl`，不需要在PolyScope手动运行RTDE程序：

```bash
ur5e-infer 20260905_150221:300 --execute \
  --backend socket --smooth-alpha 0.7 --max-linear-speed 0.20 \
  --chunks 1 --no-gripper
```

确认运动后，完整运行并启用夹爪：

```bash
ur5e-infer 20260905_150221:300 --execute \
  --backend socket --smooth-alpha 0.7 --max-linear-speed 0.20 --chunks 0
```

`--smooth-alpha 0.7` 复用旧ACT的目标EMA；`1.0`关闭EMA，较小值更平滑但响应更慢。
`--max-linear-speed 0.20` 恢复旧ACT成功链路的速度上限；此前适配层的0.05 m/s会截短
模型预测的10--14 mm步长。`--chunks 0` 持续运行到按下 `Ctrl+C`。

默认关闭chunk间保持，因此chunk内执行与旧ACT成功链路一致。`--inter-chunk-hold`
可单独启用推理期间的末端位姿保持，便于独立A/B测试，不会悄悄改变基线。

第二优先级可选实验：追加 `--diffusion-steps 10`，把本机实测推理空档从约790 ms
缩到80 ms。RoboTwin基线仍保持100步；离线抽取三处观测的平均误差基本不变。这个
参数只改变推理采样，无需重新训练。

RTDE servoJ版本保留为明确的并列实验：

```bash
ur5e-infer 20260905_150221:300 --execute --backend rtde --chunks 1 --no-gripper
```

它会临时发送机器人端servoJ脚本，再用RTDE写寄存器；当前只确认脚本运行，尚未通过
已知位移证明实机跟随。两个后端不会自动切换。夹爪读取动作第14维；`--no-gripper`
仅用于隔离机械臂测试。相机和RTDE状态均在后台持续读取，不积压旧样本。
