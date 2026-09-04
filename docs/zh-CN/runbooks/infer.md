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
ur5e-infer 600 --shadow --chunks 10
```

连接双相机和只读 RTDE，只打印预测，不发送机械臂或夹爪命令。`--chunks 0` 表示运行
到 `Ctrl+C`。`600` 会自动解析为当前唯一的 `600.ckpt`，也可以传完整路径。当前真实
链路已完成2个chunk测试，稳态单次推理约0.84秒。

## 3. 真机执行

PolyScope保持 **Remote Control**。初始化会自动检查设备、低速回原位并打开夹爪：

```bash
ur5e-infer-init
```

布置场景后，第一次只执行一个chunk：

```bash
ur5e-infer 600 --execute --chunks 1
```

确认servoJ链路后，完整运行使用：

```bash
ur5e-infer 600 --execute
```

该命令自动选择项目环境、目录、`configs/lab.yaml`与RoboTwin路径，并通过socket只发送
一次 `servoj_control_loop.script`；实际状态读取和动作下发均走RTDE。默认50个chunk，
即RoboTwin原有的300 action上限；可按 `Ctrl+C` 提前停止。

每次推理保持RoboTwin原生6步输出并依次执行；每个10 Hz TCP目标在底层插值为500 Hz
RTDE servoJ设点。夹爪读取第14维；需要暂时禁用时加 `--no-gripper`。

当前平滑版本把6个动作放在同一个0.6秒时钟内连续插值，中间动作点不再降到零速度；
相机在后台持续采帧，不阻塞500 Hz设点。servoJ自身继续使用原有lookahead。chunk之间
仍会在模型计算时保持末位姿。

首轮execute实测暴露了逐动作停启；上述修正已完成，下一次仍先用 `--chunks 1` 复测。
若机械响应仍不理想，可增加并列的socket chunk后端：每个6步chunk生成一个带 `movel
r=` 的URScript程序。这会复用旧项目已验证的批量平滑思想，但属于开环后端，不与
RTDE servoJ静默切换。
