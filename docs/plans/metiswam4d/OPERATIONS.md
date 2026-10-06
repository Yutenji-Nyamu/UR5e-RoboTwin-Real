# Metis 首版操作链

更新：2026-10-06。源码和软件接口已实现；正式权重训练、真实模型shadow及现场执行尚未验收。
环境使用 `RoboTwinSimReal`，无需重装RoboTwin或维护Gitee。`ur5e-metis`与
`python -m ur5e_real.adapters.metiswam4d`等价；机器人客户端不导入Torch/JAX。

## 数据与权重

```bash
conda activate RoboTwinSimReal
ur5e-metis prepare --data-root /data/robotics/ur5e-real \
  --runs 20261005_172729 20261005_173119 --task block_drawer_close \
  --validation-runs 20261005_173119 \
  --output outputs/metiswam4d/block_drawer_close_runtime_v1
```

该输出已存在，本机重跑请换新目录。318训练窗口、244验证窗口，10Hz/H50，末次close和padding mask保留。
不传 `--validation-runs` 时全部作为训练数据，适合最初过拟合；正式评估应保持整条轨迹留出。
图像不复制；搬迁后在 `train / run` 加 `--data-root` 指向恢复的原始数据根目录。

官方资产已经核对版本、文件大小和Alpha张量头部，未下载权重正文：

| 来源 | 所需文件 | 大小 |
| --- | --- | --- |
| [OpenWAM Alpha Foundation](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model) | checkpoint_step_154000.safetensors | 24.81GB |
| [Wan2.2 TI2V VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/tree/main/vae) | vae/config.json + diffusion_pytorch_model.safetensors | 2.82GB |

可按已核对的版本下载到忽略目录（下面下载未执行；Alpha文件还包含本基线不用的text encoder等权重）：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model",
    revision="52df4e66c82c5c8b480adcc8d01f4db7415dfb56",
    allow_patterns=["checkpoint_step_154000.safetensors"],
    local_dir="outputs/model-assets/alpha",
)
snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    revision="b8fff7315c768468a5333511427288870b2e9635",
    allow_patterns=["vae/*"], local_dir="outputs/model-assets/wan",
)
PY
```

## 训练与模型服务

```bash
ur5e-metis train --dataset outputs/metiswam4d/block_drawer_close_runtime_v1 \
  --alpha-checkpoint outputs/model-assets/alpha/checkpoint_step_154000.safetensors \
  --vae outputs/model-assets/wan/vae --steps 1000 --device cuda \
  --output outputs/metiswam4d/action_v1

ur5e-metis serve --checkpoint outputs/metiswam4d/action_v1/policy.pt --port 8006
```

仅训练Action/proprio；Video/VAE冻结，Track/未来Video缺席。一个checkpoint对应一个任务。
服务预热完成后打印 `METIS READY` 和instance ID。输入两路RGB + joint7，输出50×joint7绝对关节目标；
归一化、slot mask、当前图像VAE与训练一致。VAE搬迁可加 `--vae`，内容哈希必须一致。
当前train结束才保存，尚无定期保存/resume或全验证集评估命令；先短跑确认显存与更新，再补长跑恢复。

## 离线、shadow、现场执行

另一个终端激活同一环境，先离线测单窗口（不会连接硬件）：

```bash
ur5e-metis run --dataset outputs/metiswam4d/block_drawer_close_runtime_v1 \
  --index 318 --output outputs/metiswam4d/validation_first.json
```

index318是这份数据的首个留出窗口。输出有效步上的joint/gripper MAE与请求耗时；
不把padding算入误差，报告文件同时保存完整动作。单窗口结果不代替完整留出评估。

现场shadow只读取相机和RTDE，不发送机器人动作，也不打开夹爪串口：

```bash
ur5e-metis run --dataset outputs/metiswam4d/block_drawer_close_runtime_v1 \
  --mode shadow --lab-config configs/lab.yaml --chunks 3
```

`configs/lab.yaml`为本机配置，不入Git。shadow夹爪默认用数据契约的初始命令态，可显式 `--shadow-gripper 0/1`。
客户端超时默认且最大1.5秒，覆盖图像发送与接收；超时关闭连接，execute会停止控制器。
同步模型推理期间底层保持上一设点，2秒watchdog不变。必须实测完整模型延迟，再决定是否需要减少采样轮数或异步设计。

先查看home目标（无硬件连接）：

```bash
ur5e-metis home --dataset outputs/metiswam4d/block_drawer_close_runtime_v1
```

现场确认回位路径与动作区域后，以下命令会实际运动；本轮未运行：

```bash
ur5e-metis home --dataset outputs/metiswam4d/block_drawer_close_runtime_v1 \
  --lab-config configs/lab.yaml --execute
ur5e-metis run --dataset outputs/metiswam4d/block_drawer_close_runtime_v1 \
  --mode execute --lab-config configs/lab.yaml --action-steps 6 --chunks 1
```

home按0.1rad/s上限回数据集起点；execute要求起点误差≤0.03rad，先通过一次live图像推理才创建执行器。
每次输出50步，执行前K步后重新观察；K默认20，首次现场示例取6。底层500Hz插值，仍走现有joint速度、跟踪和TCP检查。
`--chunks`默认1且必须有限正数，Ctrl+C/异常停止；`close → open → close`中间不会提前结束。
停止时保留最后夹爪命令，不自动释放物体。暂不做任务成功判断。

源码管理、证据及尚缺项见 [STATUS_AND_NEXT](STATUS_AND_NEXT.md)。
