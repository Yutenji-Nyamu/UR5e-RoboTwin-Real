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

DP真机推理：

```bash
ur5e-infer-init
ur5e-infer <训练时间戳>:600 --execute --backend socket
```

时间戳与epoch共同标识checkpoint，例如 `20260905_150221:600`。第一次调试加
`--chunks 1 --no-gripper`。`socket` 是旧ACT已经实际运动成功的 `speedl` 链路；
`rtde` 保留为并列实验后端。完整说明见[《DP推理》](docs/zh-CN/runbooks/infer.md)。

## Diffusion Policy 快速流程

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real

# 只把 outcome=success 的raw轨迹转换为HDF5；记下输出的 HDF5_RUN
ur5e-real convert --config configs/lab.yaml \
  --task pick_place_cube --task-config simple

# N是成功轨迹数；记下输出的 ZARR_PATH
ur5e-real process-dp HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N \
  --output ZARR_PATH --trim-static-edges

# 正式训练；默认600 epoch，在第300和600轮保存checkpoint
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N

# 先离线推理，再接真机shadow
ur5e-real infer-dp --checkpoint CHECKPOINT --episode HDF5_EPISODE --index 20
ur5e-infer 600 --shadow --chunks 10
```

完整参数见[训练](docs/zh-CN/runbooks/train.md)与[推理](docs/zh-CN/runbooks/infer.md)。

## 数据与文档

数据统一存放在4TB共享盘的 `/data/robotics/ur5e-real`。架构、硬件调试、数据管理、
训练和推理文档见 [`docs/zh-CN/README.md`](docs/zh-CN/README.md)。

Diffusion Policy 主链已打通到 HDF5、Zarr、官方训练、checkpoint离线加载和真机
shadow。真机执行可明确选择已验证基础链路的socket `speedl`或实验性的RTDE servoJ。命令见
[训练](docs/zh-CN/runbooks/train.md)与[推理](docs/zh-CN/runbooks/infer.md)。

每条raw轨迹由 `session_<RUN_ID>.json` 索引，记录任务、起止时间、时长、样本数、
备注、复核历史和 `success/failure/aborted`。图像、RTDE与夹爪文件原样保留；失败或
中止数据不会训练，但也不会自动删除。`ur5e-real sessions --config configs/lab.yaml`
可随时查看汇总。HDF5与Zarr均保留源run ID和裁剪范围，checkpoint配置再引用对应
Zarr路径；数据变更简记在 `/data/robotics/ur5e-real/DATA_LOG.md`。

执行任何运动命令前，PolyScope 必须处于 **Remote Control**，工作区必须净空，人员
必须守在急停旁。
