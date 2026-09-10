# 环境部署

[English](../../runbooks/setup.md)

完整硬件→策略验收顺序及外部资产清单见[新机器部署与检修](new_machine.md)。
本页负责软件安装，不自动恢复原始数据、训练权重或工位标定。

仓库保存命令实现和依赖版本；新机器仍需安装系统驱动、创建 Conda 环境，并写入
本机配置。推荐使用 Ubuntu 24.04 LTS；当前工作站的 Ubuntu 25.04 已实机验证，但不在
RealSense 官方预编译包支持的 LTS 列表中。

## 1. 安装仓库与 Python 环境

```bash
git clone git@github.com:Yutenji-Nyamu/UR5e-RoboTwin-Real.git
cd UR5e-RoboTwin-Real
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install --no-deps -e .
scripts/bootstrap_robotwin.sh
cp configs/lab.example.yaml configs/lab.yaml
```

编辑不提交的 `configs/lab.yaml`：填写机器人地址、夹爪 `/dev/serial/by-id/...`、两台
相机序列号、home TCP和数据根目录。RoboTwin由 `robotwin.lock` 固定版本并下载到被
忽略的 `.third_party/RoboTwin`，不需要手工复制。

已有环境更新：

```bash
conda env update -n RoboTwinSimReal -f environment.yml --prune
conda activate RoboTwinSimReal
python -m pip install --no-deps -e .
```

## 2. 安装简洁指令

`pyproject.toml` 定义 `ur5e-collect`、`ur5e-replay`、`ur5e-infer`，以及
`ur5e-pi05-infer-init`/`ur5e-pi05-infer` 等命令；editable
安装会在当前 Conda 环境的 `bin/` 生成入口。只在已激活环境中使用时，到这里即可。

若希望未激活 Conda、任意目录都能直接输入这些命令，再执行一次：

```bash
scripts/install_operator_commands.sh
```

脚本会重新做editable安装，并把所有入口软链接到 `/usr/local/bin`。代码仍来自本仓库，
Python和依赖来自 `RoboTwinSimReal`；移动仓库或重建环境后重新运行脚本即可。这三个层次
分别是：

```text
/usr/local/bin/ur5e-*  ->  Conda环境/bin/ur5e-*  ->  本仓库 src/ur5e_real
```

## 3. 系统与现场状态

- 安装 RealSense 系统包和udev规则，用户加入 `dialout`；详见
  [软硬件前置条件](../PREREQUISITES.md)。
- 需要共享数据盘时，单独配置 `/data`；这不是Python包的一部分，详见
  [工作站存储](../STORAGE.md)。
- 在 PolyScope 启用并选择 Remote Control；完整RTDE依赖见
  [RTDE读写链路](../RTDE_STACK.md)。

部署后验证：

```bash
ur5e-real doctor --config configs/lab.yaml
ur5e-real doctor --config configs/lab.yaml --hardware
```

第一条检查环境、配置和数据目录；设备开机接线后，第二条再检查网络、串口和双相机。
