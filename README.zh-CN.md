# UR5e RoboTwin Real

[English](README.md)

B81L UR5e 真机的数据采集、策略训练、评估与部署基础设施：从底层设备驱动到
RoboTwin 策略适配的一站式流程。本仓库负责真机与数据，RoboTwin 保持为固定版本的
上游策略框架。

## 日常操作

以下命令会自动选择本仓库和 `RoboTwinSimReal` 环境，无需手动 `cd` 或激活 Conda。

采集：

```bash
ur5e-collect-init
ur5e-collect <任务名> --note "可选的场景备注"
```

采集中按 `c` 闭合夹爪、`o` 打开夹爪、`q` 结束；随后输入 `s`、`f` 或 `a`，标记
成功、失败或中止。记下终端显示的 `[RUN]` 编号；时间戳由程序自动生成，无需填写。

重播：

```bash
ur5e-replay-init
ur5e-replay <RUN_ID> --execute
```

初始化完成后，先恢复录制时的场景，再执行重播。完整固定流程和检查项见
[《采集与重播速查》](docs/zh-CN/runbooks/operator_workflows.md)。

## 数据与文档

数据统一存放在4TB共享盘的 `/data/robotics/ur5e-real`。架构、硬件调试、数据管理、
训练和推理文档见 [`docs/zh-CN/README.md`](docs/zh-CN/README.md)。

Diffusion Policy 主链已打通到 HDF5、Zarr、官方训练、checkpoint离线加载和真机
shadow；命令见[训练](docs/zh-CN/runbooks/train.md)与[推理](docs/zh-CN/runbooks/infer.md)。

执行任何运动命令前，PolyScope 必须处于 **Remote Control**，工作区必须净空，人员
必须守在急停旁。
