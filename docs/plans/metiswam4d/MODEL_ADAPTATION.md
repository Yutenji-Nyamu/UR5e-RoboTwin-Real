# MetisWAM4D：单臂 RGB → Action 首版

更新：2026-10-06。采用用户确认的 π0.5 关节约定。实现位于
[adapter](../../../src/ur5e_real/adapters/metiswam4d/__main__.py)，使用本项目跟踪的 `metiswam4d/` 原生模型源码。

## 当前输入、输出和更新范围

| 项目 | 首版约定 |
| --- | --- |
| 物理状态 | `[actual_q_0..5, commanded_gripper]`，关节 rad；夹爪 0=open/1=closed |
| 视觉 | head/wrist RGB，各缩放 320×192，上下堆叠；Wan VAE 单个当前帧 latent，`(mode - mean) / std` |
| 模型输入 | clean video `[B,48,1,24,20]`（匹配的 Wan2.2 TI2V VAE），状态 `[B,1,80]`，含噪动作 `[B,50,80]` |
| 动作内部表示 | 每个未来 q 相对当前 q 的 delta，夹爪保持绝对命令态；训练集 mean/std，std 下限 1e-3 |
| 80D 有效位置 | **10–16** 对应单臂 joint7，其他 73 维固定零且不计 loss；不复制成第二只手臂 |
| 输出 | 10 Hz、50 步 `[q6绝对目标, g]`；可包装 `[q6,0,q6,g]` 为现有 joint14 |
| 文本 / embodiment | 首版省略；**一个 checkpoint 对应一个任务**，不声称支持任意语言指令 |
| Track / depth / mask / camera motion | 分支缺席，不复制 RGB 冒充，不依赖新增深度才能训练此基线 |
| Video | 仅当前图像的 clean tokens，参数冻结；无未来帧，无 Video loss |
| 更新参数 | `action.*` 与 `proprio_encoder.*`；其余冻结，优化器只含这两组 |
| 采样 | 16 轮默认 Euler flow matching，sigma 从 1 到 0；输出夹爪裁剪到 [0,1]，关节不静默裁剪 |

上游 `attention.py` 在 `action_read="none"` 时仍允许 Action 读取 clean Video tokens；
这里关闭的是未来世界读取，不是当前图像。原 `AsyncSampler` 仅有未来视频时才送 Video，
所以首版使用独立的 Action 采样循环，每轮始终送入当前 RGB 条件。
原 H32/8 的约束属于上游视频/Track 数据窗口，当前帧 Action-only adapter 不调用该窗口校验，
H50 已在原生模型训练与采样中验证。

这是一条临时适配基线。Alpha 的预训练动作原本是 EEF 槽位，改用我们的 joint7 后需要微调验证；
没有把 EEF 结果冒充关节角。当前数据只有两条 success，不据此声称学到了可部署策略。

## 数据转换：保留完整任务

```bash
conda activate RoboTwinSimReal
python -m ur5e_real.adapters.metiswam4d prepare \
  --data-root /data/robotics/ur5e-real \
  --runs 20261005_172729 20261005_173119 \
  --task block_drawer_close \
  --output outputs/metiswam4d/block_drawer_close_v1
```

该目录已在本机生成，重跑须换一个不存在的输出目录。

- 仅接受显式选择的、task 一致、completed + reviewed success、v3 实测关节数据；不使用 TCP-only。
- 10 Hz 重采样：关节线性插值，夹爪零阶保持，图像最近邻且平局优先早帧。
  源时间必须递增，最大空隙 0.25 秒、图像偏移最大 0.1 秒。误差和原始哈希写入审计。
- 不按 close/open 周期裁剪，不假定最终 open；全部有序原始夹爪事件随审计保存。
  二值标签仍不表示“重复按 close”的独立动作或真实夹爪宽度。
- 时刻 t 的 action 是 t+0.1 秒开始的未来实测状态；末帧只作 next-state 标签。
  episode 尾部重复末个目标补 H50，并用 valid mask 排除 padding loss 和统计。
- 数据集保存 `actions.npz`（state/action/valid）、`images.json`（原始图像引用）、`contract.json`
  （动作/图像/归一化约定）、`audit.json`。原始数据和图像不改写、不复制。
  图像引用在本机是绝对路径，搬到别的主机需重新生成索引。

实际两条轨迹的网格分别 319/245 点，产生 318/244 = **562** 个窗口。
窗口数与原 RGB 帧数不同来自显式重采样，最大图像偏移 100 ms，见
[转换审计](evidence/dataset_conversion_20261006.json)。全部 close → open → close 均保留。

## 可复现的结构测试

```bash
python -m ur5e_real.adapters.metiswam4d smoke \
  --dataset outputs/metiswam4d/block_drawer_close_v1 \
  --device cuda --dtype bfloat16 --output outputs/metiswam4d/new_smoke
```

CPU 默认也可跑。使用**原生 tiny Video/Action**，对真实首个训练窗口做 8 次更新、4 轮采样，
检查冻结哈希相同、Action/proprio 更新、图像扰动改变输出、保存重载逐元素相同、输出 joint7/joint14。
CUDA 测试启用 gradient checkpointing。审查后改为 Action/proprio 参数及优化器状态保持FP32，
BF16 autocast计算；已用默认1e-5学习率重新验证更新及精度保持的保存重载。已执行 CPU FP32 和 RTX A6000 BF16 两种测试。

测试图像编码是明确标记的像素池化，仅验证计算链路，**不是预训练 VAE 或正式模型效果**。
`infer` 默认拒绝 tiny checkpoint。loss 波动和一次窗口 smoke 不作为收敛/成功率结论。

## 正式权重到位后的入口

以下是已实现但**本轮未执行成功的 production 路径**，因为本机尚缺模型和 VAE 权重：

```bash
python -m ur5e_real.adapters.metiswam4d train \
  --dataset outputs/metiswam4d/block_drawer_close_v1 \
  --alpha-checkpoint /path/to/checkpoint_step_N.safetensors \
  --vae /path/to/Wan2.2-TI2V-5B/vae \
  --steps 1000 --lr 1e-5 --device cuda \
  --output outputs/metiswam4d/action_v1

python -m ur5e_real.adapters.metiswam4d infer \
  --checkpoint outputs/metiswam4d/action_v1/policy.pt \
  --head /path/to/head.png --wrist /path/to/wrist.png \
  --state Q0 Q1 Q2 Q3 Q4 Q5 GRIP \
  --device cuda --output outputs/metiswam4d/prediction.npz
```

路径和 Q0…GRIP 是实际文件/数值占位符。原生 Alpha loader 严格检查 Video/Action/proprio 张量；
无权重不静默随机初始化。Wan VAE 仅从指定本地目录加载，不后台下载。
若拿到的是 Metis DCP，应补该格式 loader，不能把它直接传给 `--alpha-checkpoint`。

环境沿用 `RoboTwinSimReal`：torch 2.4.1+cu121、diffusers 0.34.0（已核实含 AutoencoderKLWan）、
NumPy/OpenCV；Alpha 加载另需 safetensors。未导入上游依赖 pyarrow 的通用训练/data 启动器。
本轮不更换现有环境。production 参数规模、显存峰值、VAE 输出 shape 和速度需在正式权重上确认。

训练是单任务、batch=1 的首版循环，保存完整模型、源码 SHA256、配置、数据契约及统计；
冻结哈希训练前后对比。离线推理复用同一编码/归一化/解码，检查本地源码与 checkpoint 的哈希一致。
VAE 目录由 checkpoint 记录，搬迁需保留该资产；暂未做通用 resume、分布式训练或服务化。

## 真机边界

此版本 `infer` 只产生 `.npz`，不发送运动。之后按现有 π0.5 分层接入：

```text
双 RGB + actual_q/g → Metis 独立策略进程 → 50×joint7
    → 关节/速度/TCP 边界检查 → 执行前 K 步 → 新观测重规划
```

外部 joint14 包装已具备；机器人应复用现有 joint chunk / 500 Hz executor，
不从模型模块直接调用硬件。K、每次观测到 chunk 的延迟、允许过期时间要实测。
“每个时间步”仍是关节目标序列，模型按 chunk 滚动预测，不是 500 Hz 每拍跑一次大模型。

`close → open → close` 没有额外训练障碍；旧 π0.5 的周期校验和 open 后自动结束属于应用规则，
首版转换已解除该裁剪假设。未来新 executor 需明确人工/超时/任务判定结束，不能在中途 open 时结束。
未完成正式权重评估、shadow 延迟测试、服务协议与现场执行之前，不宣称已接通真机策略。
