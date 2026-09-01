# UR5e RoboTwin Real

[English](README.md)

B81L UR5e 真机的数据采集、策略训练、评估与部署基础设施：从单设备底层轮子，
到 RoboTwin 策略集成的一站式、可复用流程。

```text
UR5e / 夹爪 / RealSense
          ↓
硬件层 → 控制层 → 采集与数据层 → 策略适配层
                                  ↓
                         固定版本的上游 RoboTwin
```

本仓库负责真机设备、时间同步、数据契约、安全限制和部署适配。RoboTwin 保持为
固定版本的独立上游框架；策略代码不能直接控制 socket、串口或相机。

## 开始

```bash
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install -e .
cp configs/lab.example.yaml configs/lab.yaml
ur5e-real doctor --config configs/lab.yaml --hardware
```

当前工作站的原始数据与转换数据统一放在4TB共享盘的
`/data/robotics/ur5e-real`。

## 主要目录

- `examples/smoke/`：有边界的单设备检查。
- `src/ur5e_real/hardware/`：RTDE、URScript、串口夹爪、RealSense。
- `src/ur5e_real/control/`：轨迹与 RTDE servoJ 流式控制。
- `src/ur5e_real/collection/`、`data/`：同步采集与数据转换。
- `src/ur5e_real/adapters/`：窄接口的 RoboTwin 策略适配。
- `robot_programs/`：PolyScope/RTDE 机器人端程序。
- `integrations/robotwin/`：上游版本信息与经过审查的小补丁。

ACT 是当前真机基线，Diffusion Policy 是下一目标。除非实测证据明确要求，否则
沿用 RoboTwin 原始模型配置。socket 手动重播保持为独立的验机链路。

## 文档

架构、真机调试、存储与数据管理、迁移记录和操作手册见
[`docs/zh-CN/README.md`](docs/zh-CN/README.md)。英文版本见
[`docs/README.md`](docs/README.md)。

## 仓库原则

本机配置、原始数据、视频、checkpoint、缓存和上游工作树均不提交。旧仓库保持
不变。可能引起真机运动的命令必须有显式执行步骤，并由人员守在急停旁。
