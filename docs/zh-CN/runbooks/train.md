# RoboTwin Diffusion Policy 训练

[English](../../runbooks/train.md)

先进入统一环境和仓库：

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
```

## 1. 从成功轨迹构建训练数据

```bash
ur5e-real convert --config configs/lab.yaml \
  --task pick_place_cube --task-config simple
```

记下输出的 `HDF5_RUN` 路径；`N` 是纳入的成功轨迹数。再生成 RoboTwin DP Zarr：

```bash
ur5e-real process-dp HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N \
  --output /data/robotics/ur5e-real/converted/dp/DATA_VERSION.zarr \
  --trim-static-edges
```

Zarr 默认写到 `/data/robotics/ur5e-real/converted/dp/`。它使用头部相机、14维
TCP/夹爪状态，以及 `action[t] = state[t+1]`。`--trim-static-edges` 仅裁每条轨迹
最前、最后的连续静止区：默认阈值2 mm/1°，两端各保留3帧，不裁中间停顿；源run ID、
原长度和实际裁剪范围均写入Zarr属性。文件名应包含轨迹数、裁剪规则和HDF5时间戳；
不使用 `--overwrite`，因此裁前裁后版本并存。

## 2. 冒烟训练

```bash
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N \
  --debug --batch-size 8
```

`--debug` 沿用 RoboTwin 模型，只把训练缩为2个epoch、每个epoch最多3步。单轨迹时
验证集比例自动设为0。

## 3. 正式训练

```bash
ur5e-real train-dp ZARR_PATH \
  --task pick_place_cube --task-config simple --episodes N
```

不加 `--debug` 即沿用 RoboTwin `robot_dp_14.yaml`：horizon 8、3步观测、6步动作、
10 Hz和600 epoch。batch默认不超过128，并在小数据集上自动缩到最短episode以内，
保证留出的验证轨迹至少产生一个batch。训练目录及checkpoint自动写入
`/data/robotics/ur5e-real/checkpoints/dp/`。

默认每300轮保存一次，即第300和600轮；每轮train/validation loss都写入运行目录的
`logs.json.txt`。

2026-09-04 首次正式运行纳入6条成功轨迹（869个transition），最后1条留作验证，
其余5条训练；600 epoch约15.6分钟完成，`300.ckpt`和`600.ckpt`均已生成并通过
留出轨迹离线加载。训练loss已收敛，验证loss约300轮后进入平台并在末段轻微回升，
因此先保留两个checkpoint做真机对照。2条中止轨迹保留在raw中，但未进入该数据版本。

2026-09-05 新版本纳入17条成功轨迹，源HDF5为 `run_20260905_150058`；按上述规则由
2109裁为1562 transitions。训练 `20260905_150221` 已完成600 epoch并生成两个
checkpoint。验证曲线约在250--300轮最低，之后训练loss继续下降、验证loss上升；因此
首测用 `20260905_150221:300`，`20260905_150221:600` 保留作对照。曲线为训练目录下
的 `loss_curve.png`；简短变更记录位于 `/data/robotics/ur5e-real/DATA_LOG.md`。

## ACT（保留的旧适配）

```bash
ur5e-real process-act HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N
scripts/train_act.sh pick_place_cube simple N 0 0
```

ACT 与 DP 是并列适配器；正式方向当前为 DP。
