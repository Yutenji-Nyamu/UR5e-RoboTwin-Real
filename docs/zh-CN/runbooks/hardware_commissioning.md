# 硬件逐项调试

[English](../../runbooks/hardware_commissioning.md)

按以下顺序自底向上排错。第2–3节的软件检查只读；第1节的开机/解除制动须现场确认。
从系统安装到资产迁移和集成验收的总顺序见[新机与检修手册](new_machine.md)。

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
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10 --joints
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
5. 按推理手册运行默认RTDE servoJ策略；socket仅保留作对照。

不要把首次夹爪、freedrive、重播和 servoJ 测试合并为一次运行；每个设备必须有
独立、可理解的故障边界。

## 5. RTDE启动握手偶发失败

本次`ur5e-collect-init`报错位于`RtdeTcpClient.connect()`，在发送`move_linear()`及开爪之前。
home/moveL文字是预先打印的计划，不代表当时已经发送运动。

已安装的`UrRtde==2.7.12`等待回复默认1秒；日志先出现`no data received in last 1 seconds`，
协议请求返回空结果，随后与真正拒绝协议一样抛出`Unable to negotiate protocol version`。
因此可以确定的是**启动握手回复超时**，不能仅凭异常名称判定版本不兼容。
重新执行成功符合瞬时故障，但具体是控制器忙、连接切换时序还是网络抖动，尚无抓包证据。
UR协议将版本接受、recipe协商和启动同步区分为不同阶段。
[官方RTDE协议](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html)

2026-09-10对[RTDE客户端](../../../src/ur5e_real/hardware/rtde.py)的修复：

- 只对connect/协议/控制器版本握手最多尝试3次，间隔0.5秒；每次新对象、新连接，失败立即断开。
- 缺失控制器版本回复也视为握手失败。旧库在握手失败后仍保留socket，再对同一对象connect可能直接跳过协商；
  TCP servoJ原有的复用对象重试也改用公共入口，joint启动同样复用。
- 读客户端recipe/start异常也清理连接；不重试寄存器占用、寄存器写入、URScript发送或回原位命令。
- doctor由裸30004端口开关探测改为完整的**只读协议/版本握手**后断开。
  原探测可能增加连接切换，但没有证据断言它就是此次超时的原因。
- 不改底层流超时、运行中掉线即停止的行为或servoJ watchdog。
  耗尽后init打印`[BLOCKED] RTDE startup failed ...`；握手成功不等于500Hz执行链已验收。

已用故障注入/回归测试验证重试、清理、Ctrl-C和失败不发home/开爪；本轮没有连接硬件复现瞬时故障。
如果仍连续失败，先运行有限次数只读RTDE smoke并保留报错与PolyScope状态；检查有线链路/IP冲突、
RTDE服务、控制器负载及并发客户端。真正的协议/recipe不兼容仍需修正；不自动降协议或关闭保护，
也不反复重发运动。`input register in use`是后续寄存器所有权问题，和本次超时不是同一故障。
