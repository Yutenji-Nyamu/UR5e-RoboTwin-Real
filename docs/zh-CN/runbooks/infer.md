# RoboTwin DP 推理

[English](../../runbooks/infer.md)

先进入统一环境和仓库：

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
ur5e-real infer-dp --checkpoint CHECKPOINT \
  --config configs/lab.yaml --shadow --chunks 10
```

连接双相机和只读 RTDE，只打印预测，不发送机械臂或夹爪命令。`--chunks 0` 表示运行
到 `Ctrl+C`。当前真实链路已完成2个chunk测试，稳态单次推理约0.84秒。

## 3. 真机执行

先在 PolyScope **Remote Control** 下初始化：

```bash
ur5e-policy-init
```

然后把 PolyScope 切到 **Local**，启动由
`robot_programs/servoj_control_loop.script` 创建的机器人端程序，再执行一个chunk：

```bash
ur5e-real infer-dp --checkpoint CHECKPOINT \
  --config configs/lab.yaml --execute --chunks 1
```

每次推理保持 RoboTwin 原生6步输出并依次执行；每个10 Hz TCP目标在底层插值为
500 Hz RTDE servoJ设点。夹爪读取第14维；需要暂时禁用时加 `--no-gripper`。

离线和shadow已经实测；`--execute`代码已接好，下一次现场从servoJ保持和毫米级小步
验证开始，再接正式训练模型。
