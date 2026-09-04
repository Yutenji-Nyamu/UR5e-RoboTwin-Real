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
10 Hz、batch 128和600 epoch。训练目录及checkpoint自动写入
`/data/robotics/ur5e-real/checkpoints/dp/`。

2026-09-04 已用一条真实轨迹完成原生 Dataset 读取、2 epoch GPU训练并生成两个
可重新加载的checkpoint。

## ACT（保留的旧适配）

```bash
ur5e-real process-act HDF5_RUN \
  --task pick_place_cube --task-config simple --episodes N
scripts/train_act.sh pick_place_cube simple N 0 0
```

ACT 与 DP 是并列适配器；正式方向当前为 DP。
