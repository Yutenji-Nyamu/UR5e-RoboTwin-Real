# 硬件逐项调试

[English](../../runbooks/hardware_commissioning.md)

按以下顺序自底向上排错。第3节及以前全部只读，不会运动机械臂或夹爪。

## 当前 B81L 基线

2026-09-01 实测：

- 工作站与UR5e处于同一静态子网，实际地址见本机 `configs/lab.yaml`；
- Dashboard `29999`、URScript `30001`、RTDE `30004` 均连通；
- PolyScope `5.13.0`，机器人 `RUNNING`、安全状态 `NORMAL`、当前Local控制，
  无程序运行；
- 夹爪：CH340 串口转换器，使用稳定的 `/dev/serial/by-id/...` 路径；
- 配置中的头部与腕部 D435i 均可同时读取 `640x480` 彩色帧。

## 1. 开机与 PolyScope

1. 松开物理急停，确认工作区无人、无障碍物。
2. 按示教器电源键，等待 PolyScope 启动。
3. 在初始化界面依次按 **ON（开机）**、**START（启动）**；状态应变为
   `RUNNING`，安全状态保持 `NORMAL`。
4. 确认网线连接，机器人地址与 `configs/lab.yaml` 一致。
5. 下面的只读检查不需要加载或运行运动程序。

应先在 PolyScope 设置中启用并保持 Remote Control。外部 URScript、freedrive、
手动重播和DP执行均使用Remote模式；`ur5e-infer` 自动启动机器人端servoJ程序，
动作目标仍由RTDE发送。

## 2. 机器人只读检查

```bash
python examples/smoke/polyscope_status.py --config configs/lab.yaml
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10
```

预期：Dashboard 返回 `RUNNING/NORMAL`；RTDE 以10 Hz输出稳定 TCP 位姿。这些读取
不需要 PolyScope 程序运行。

## 3. USB 检查

```bash
python examples/smoke/realsense.py --config configs/lab.yaml --frames 3
ur5e-real doctor --config configs/lab.yaml --hardware
```

若某个 RealSense 序列号缺失，重新插紧相机两端的数据线，并优先使用主机后部
USB 3.x 接口。当前夹爪协议没有身份/状态回读，因此 doctor 只能证明目标串口存在。

D435i 刚启动的彩色帧可能偏暗、严重偏色，这是自动曝光和自动白平衡尚未收敛。
[librealsense官方OpenCV示例](https://github.com/realsenseai/librealsense/blob/master/doc/stepbystep/getting_started_with_openCV.md)
也会丢弃启动帧。本实验室实测白平衡比曝光收敛更晚，因此公共相机封装先丢弃60帧
（30 FPS时约2秒）再返回数据；只有在实测现场光照后才调整 `warmup_frames`。

## 4. 第一次输出检查

以下操作需要人员守在现场，并显式添加 `--execute`：

1. 将机械臂停在夹爪四周无障碍的位置；目视确认后只执行一次 `open`，再执行一次
   `close`。
2. 用 `ur5e-real prepare --config configs/lab.yaml` 检查原位目标，现场确认后加
   `--execute` 验证低速回原位及打开夹爪。
3. 扶住机械臂，分别测试 freedrive 开启与停止。
4. 录制一段短轨迹；先 dry-run；再只执行一个重播段。
5. 按推理手册运行一个仅机械臂socket策略chunk；RTDE servoJ另用已知位移测试。

不要把首次夹爪、freedrive、重播和 servoJ 测试合并为一次运行；每个设备必须有
独立、可理解的故障边界。
