# 软硬件前置条件与分层调试

[English](../PREREQUISITES.md)

本文替代旧 README 中零散的安装笔记，只保留本项目实际需要的组件、用途和最短调试
入口。

## 当前工作站基线

| 层 | 当前配置 | 用途 |
|---|---|---|
| 系统 | Ubuntu 25.04 | RealSense、USB串口和UR网络主机 |
| GPU | NVIDIA RTX A6000 48 GB，驱动 `580.95.05` | DP训练与推理 |
| Python | Conda `RoboTwinSimReal`，Python `3.10.18` | 统一真机与策略环境 |
| PyTorch | `2.4.1+cu121`，CUDA可用 | RoboTwin模型运行 |
| 数据盘 | 4 TB Seagate NTFS3，自动挂载到 `/data` | raw、转换数据、checkpoint和日志 |
| RoboTwin | `.third_party/RoboTwin`，固定提交 `21072034...` | 上游模型与训练代码 |

2026-09-04 已修复NTFS的MFT镜像与dirty标记，systemd自动挂载恢复正常；当前剩余
约3.0 TiB。该故障来自卷未干净卸载，不是权限检查或按需挂载设计造成的。

## 真机一次性准备

### UR5e 与 PolyScope

- 旧项目记录B81L现场需要开启5个断路器；启动后在PolyScope执行 **ON** 和
  **START**，机器人应为 `RUNNING/NORMAL`。
- 机器人与工作站有线网口使用同一静态子网；实际地址只保存在不提交的
  `configs/lab.yaml`。
- Dashboard `29999` 用于状态，URScript `30001` 用于初始化/采集/手动重播，RTDE
  `30004` 用于状态和servoJ设点。
- `Remote Control`：初始化、freedrive、采集、socket重播。
- `Local`：加载并运行机器人端RTDE servoJ程序，然后由电脑发送设点。

当前仓库保留可读的
`robot_programs/servoj_control_loop.script` 和 RTDE XML。旧项目另有已经使用过的
`translation_sample_servoj.urp`；下一次servoJ调试时现场加载/确认一次，再决定是否
把导出的URP作为机器人资产纳入新仓库。

### 夹爪

- 打开夹爪电源，USB串口连接电脑；
- 用户需属于 `dialout`；
- 配置使用 `/dev/serial/by-id/...` 稳定路径，不依赖易变化的 `/dev/ttyUSB0`；
  实际设备名只保存在 `configs/lab.yaml`。

### RealSense

- 头部和腕部各一台D435i，串号映射只保存在 `configs/lab.yaml`；
- 两台优先连接主机后部USB 3.x端口；
- 系统组件：`librealsense2-dkms`、`librealsense2-utils`、
  `librealsense2-dev`、`librealsense2-udev-rules`；
- Python使用 `pyrealsense2==2.56.5.9235`；启动后丢弃60帧，让自动曝光和白平衡
  收敛。

相机独立查看工具：

```bash
realsense-viewer
```

## 软件安装

### 本仓库与基础环境

```bash
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install -e .
cp configs/lab.example.yaml configs/lab.yaml
scripts/bootstrap_robotwin.sh
```

已有环境更新时使用 `conda env update -n RoboTwinSimReal -f environment.yml`。本机
`configs/lab.yaml` 不提交，保存机器人地址、相机序列号、夹爪路径、原位和数据根。

### DP附加依赖

`environment.yml` 已锁定完整DP运行栈，其中上游额外需要：

- `hydra-core==1.2.0` 与 `omegaconf==2.3.0`：读取上游训练配置；
- `numba==0.61.2`：Zarr序列批采样；
- `dill==0.3.8`：保存和加载DP checkpoint。

这些版本已安装到 `RoboTwinSimReal` 并通过训练与checkpoint加载。

不需要为本项目安装ROS，也不需要把完整RoboTwin复制进仓库。当前内核和
librealsense udev规则已经识别CH340与两台相机，无需再编译旧CH341驱动副本。

## 最短分层调试

| 要确认的层 | 命令/动作 | 通过标准 |
|---|---|---|
| Python与配置 | `ur5e-real doctor --config configs/lab.yaml` | Python模块、XML和数据根正常 |
| 全部只读硬件 | 加 `--hardware` | UR端口、串口、两个相机均为 `[OK]` |
| 单相机画面 | `python examples/smoke/realsense.py --config configs/lab.yaml --frames 3` | 串号映射正确、预热后画面正常 |
| RTDE读 | `python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10` | 10 Hz TCP连续输出 |
| 采集/重播 | 见[采集与重播速查](runbooks/operator_workflows.md) | 已完成一条端到端实机验证 |
| DP离线 | 见[训练手册](runbooks/train.md) | 转换、batch、短训练和checkpoint重载已通过 |
| servoJ | PolyScope Local运行URP，再做保持/小步/序列 | 实测TCP跟随连续 |
| DP在线 | 见[推理手册](runbooks/infer.md) | shadow已通过；下一步验证servoJ后执行 |

当前下一步是servoJ保持、毫米级小步和6步目标序列。
