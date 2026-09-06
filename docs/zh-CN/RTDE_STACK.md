# RTDE读写链路

[English](../RTDE_STACK.md)

## 当前实现

RTDE“读”和“写”不是同一件事：Python可直接从 `30004` 订阅机器人状态；执行位姿时，
Python把目标写入RTDE输入寄存器，同时机器人控制器上必须有URScript读取寄存器并调用
`servoJ`。本项目会自动发送该脚本，无需手动加载URP。

```text
ur5e-infer
├─ 相机 + DP模型：产生6步TCP动作
├─ 30001：发送 robot_programs/servoj_control_loop.script
└─ 30004：UrRtde客户端以500 Hz写输入寄存器
                  ↓
     控制器URScript读取目标、逆解并调用servoJ
                  ↓
                UR5e
```

| 层 | 仓库内内容 | 仓库外要求 |
|---|---|---|
| Python协议 | `UrRtde==2.7.12` 固定在 `environment.yml` | Conda环境已创建 |
| 状态读取 | `src/ur5e_real/hardware/rtde.py` | UR5e的 `30004` 可达 |
| 设点流 | `src/ur5e_real/control/servoj.py` | 没有其他客户端占用相同输入寄存器 |
| RTDE字段 | `robot_programs/control_loop_configuration.xml` | 控制器支持RTDE |
| 机器人循环 | `robot_programs/servoj_control_loop.script` | PolyScope处于Remote Control且机器人可运行 |
| 策略适配 | `src/ur5e_real/adapters/robotwin_dp/` | 固定版本RoboTwin和checkpoint |

当前链路不依赖ROS、External Control URCap或示教器中手动播放的URP。旧项目中
`translation_sample_servoj.urp` 的Local模式流程是历史实验，不是当前DP前置条件。

## PolyScope与网络要求

1. 在 PolyScope 的 **Settings → System → Remote Control** 启用远程控制。
2. 从右上角控制配置中选择 **Remote Control**，不能停留在Local。
3. 机器人执行 **ON**、**START**，状态应为 `RUNNING`、安全状态为 `NORMAL`。
4. 工作站与UR5e处于同一有线子网；实际地址写在不提交的 `configs/lab.yaml`。
5. 允许 `29999`（Dashboard）、`30001`（URScript）和 `30004`（RTDE）。

UR官方说明中，e-Series的RTDE目标频率为500 Hz；Remote Control需先在设置中启用，
再从控制配置中选择。参见
[RTDE手册](https://www.universal-robots.com/manuals/EN/HTML/SW10_10/Content/Prod-RTDE/Real_Time_Data_Exchange_RTDE.htm)、
[客户端端口说明](https://www.universal-robots.com/articles/ur/interface-communication/remote-control-via-tcpip/)和
[Remote Control设置](https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/prod-usr-man/software/PolyScope/content/hamburger_menu_g5/System_remote_en.htm)。

## 最短验证顺序

```bash
# 环境、端口、串口和双相机枚举
ur5e-real doctor --config configs/lab.yaml --hardware

# 只读：直接订阅实际TCP，不要求机器人程序正在播放
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10

# 只推理，不向机器人发送动作
ur5e-infer 20260905_150221:600 --shadow --chunks 10

# 实机：自动发送URScript并启动RTDE servoJ流
ur5e-infer-init
ur5e-infer 20260905_150221:600 --execute
```

“RTDE能读但不能动”本身不矛盾：读取只需连接，运动还需要Remote Control、正在运行的
机器人端servoJ脚本和输入寄存器流。当前默认参数是已实机完成任务的10步DP去噪、
500 Hz流、`lookahead=0.1`、`gain=300`；需要对照时才显式切换 `--backend socket`。
