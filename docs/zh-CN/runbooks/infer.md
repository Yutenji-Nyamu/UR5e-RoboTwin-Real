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

布置场景后，从以下三个互不覆盖的执行版本中选择。

### A. socket基线（已验证chunk内平滑）

它用RTDE读状态、Socket 30001发送 `speedl`，不需要在PolyScope手动运行RTDE程序：

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition baseline \
  --smooth-alpha 0.7 --max-linear-speed 0.40 --chunks 0
```

这是此前成功完成任务的基线。`--smooth-alpha 0.7` 是chunk内目标EMA；
`--socket-transition baseline` 保持原控制逻辑，不处理推理空档。

### B. socket末步拉伸（本次新增）

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend socket --socket-transition stretch \
  --smooth-alpha 0.7 --max-linear-speed 0.40 \
  --diffusion-steps 10 --chunks 0
```

前五个动作与基线完全相同。只把最后一个目标均匀摊到
`100 ms + 本次推理耗时 + 20 ms`，下一chunk到达后直接接管；不使用后台线程，也不并发
写30001。终端会打印每个chunk实际采用的 `final speedL` 时长。

`--diffusion-steps 10` 在本机把推理约从790 ms降到80 ms；此前三处离线样本的平均误差
基本不变。该参数仅改变推理采样，无需重训。

### C. RTDE 500 Hz servoJ

模型、相机和观测不变，只替换执行层：

```bash
ur5e-infer 20260905_150221:600 --execute \
  --backend rtde --max-linear-speed 0.40 --diffusion-steps 10 \
  --servoj-lookahead 0.1 --servoj-gain 300 --chunks 0
```

程序自动发送机器人端servoJ循环，把6个10 Hz目标插值成500 Hz设点；推理期间持续保持
最后设点，因此没有socket命令超时式硬刹，但仍可能有短暂停顿。`0.1/300` 是UR官方
servoJ默认值；可在 `lookahead=0.03--0.2`、`gain=100--2000` 范围内显式调整。
此后端代码已接通，但尚待本仓库第一次实机跟随验证；可先把末尾改为
`--chunks 1 --no-gripper` 做一次对照。

三个版本都不会自动互相切换。`--chunks 0` 持续运行到 `Ctrl+C`；夹爪读取动作第14维，
`--no-gripper` 仅用于隔离机械臂。
