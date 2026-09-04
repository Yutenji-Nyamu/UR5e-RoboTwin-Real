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
  --task pick_place_cube --task-config simple --episodes N
```

Zarr 默认写到 `/data/robotics/ur5e-real/converted/dp/`。它使用头部相机、14维
TCP/夹爪状态，以及 `action[t] = state[t+1]`。

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

## ACT（保留的旧适配）

```bash
ur5e-real process-act HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N
scripts/train_act.sh pick_place_cube simple N 0 0
```

ACT 与 DP 是并列适配器；正式方向当前为 DP。
