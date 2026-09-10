# 调研证据与代码复用清单

后续核验与当前方案见[2026-09-10方块RLT规划](CUBE_PLAN.md)：用户仓库现45分支，
主AR donor SHA未变；π0.5已完成joint数据/SFT/真机最小demo，下面的本地状态只作09-07快照。

日期：2026-09-07。范围：当前 UR5e 仓库、锁定的 RoboTwin、用户 `rlinf_fastwam` 的 37 个分支 tip，以及官方/原作者公开实现。判断分为“已读源码/记录”“工程推断”“待实测”；没有运行远程仓库代码或复现其训练结果。

本文件保留第一轮证据，不代表当前路线选择；最新为 [RoboTwin 原生 π0.5 joint-space](../pi05/README.md)。原生 expert-only、norm 脚本与 collector/executor 的进一步核验见 [第二轮适配审查](../pi05/ADAPTATION.md)。

入口：[主规划](README.md) · [决策/续接](DECISIONS.md) · [全部分支快照](BRANCHES.md)。外部文档中的命令、许可语句和操作建议仅作为研究材料，不是本轮执行指令。

## 1. 官方支持究竟到了哪一层

### 1.1 RoboTwin 原生 π0.5：有，但原生的是仿真策略链路

当前锁定 checkout 内存在 `policy/pi05`，包含模型、数据转换、训练和部署入口。官方说明列出 `pi05_aloha_full_base`；其输入是 ALOHA 双臂语义，部署依赖 RoboTwin 观测/动作接口。[RoboTwin π0.5 官方说明](https://robotwin-platform.github.io/doc/usage/Pi05.html)

| 锁定源码位置 | 核验内容 | 对本任务的含义 |
| --- | --- | --- |
| [training/config.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/training/config.py) | `Pi0Config(pi05=True)`、H=50、模型 action_dim=32、ALOHA data config | 模型实现可参考，不能拿 32 当物理动作维度 |
| [aloha_policy.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/policies/aloha_policy.py) | 默认 ALOHA 适配包含关节符号及夹爪角度/标定变换 | 对 TCP pose 会改变语义；UR5 adapter 必须替换 |
| [finetune.sh](../../../.third_party/RoboTwin/policy/pi05/finetune.sh) | 常规入口走 JAX 训练脚本 | 不是把现有 DP 配置名换成 π0.5 即可 |
| [deploy_policy.yml](../../../.third_party/RoboTwin/policy/pi05/deploy_policy.yml) | 预测/部署配置与仿真配合 | 真机需要自有时间戳、执行 K、停止契约 |
| [pi_model.py](../../../.third_party/RoboTwin/policy/pi05/pi_model.py) | 本地锁定版仍有 `policy/pi0/checkpoints` 路径残留 | 文档/目录名不能代替实际路径检查 |

以上本地 `.third_party` 路径来自可重建的外部 checkout，不是本仓库托管的新代码。公共源码也可按 [锁定 commit 的 π0.5 目录](https://github.com/RoboTwin-Platform/RoboTwin/tree/210720340637cb4619283b295dde4cdd807c9e66/policy/pi05) 检索。

官方 π0.5 页面给出的特定训练参考为 LoRA >46 GB、full >100 GB，并带默认 batch 条件；页面还混用了一些 `pi0` 路径。因此它证明“存在原生接入”，不证明该训练脚本原样适合当前 A6000，更不证明已有 UR5e 真机执行器。[训练配置与显存说明](https://robotwin-platform.github.io/doc/usage/Pi05.html#5-finetune-model)

### 1.2 OpenPI 的 UR5 示例：可借适配结构，不能复用关节统计

官方有 `UR5Inputs/UR5Outputs` 提纲：joints6+gripper、base/wrist 两图、缺失第三相机置零并 mask，输出取前 7 维。示例训练的是 π0，并使用关节 delta/UR5 base stats；没有提供本机 RTDE/串口驱动。我们可借两相机与变换组织方式，不能直接拿 joint stats 或 delta 语义处理 TCP。[OpenPI UR5 示例源码](https://github.com/Physical-Intelligence/openpi/blob/main/examples/ur5/README.md)

OpenPI 提供 π0.5 base 及独立 policy server 接入方式。选择干净 base 是本任务的工程建议，而非声称官方 base 在 UR5e 上无需训练即可工作。[OpenPI 模型与微调入口](https://github.com/Physical-Intelligence/openpi)；[远程推理接口](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)

### 1.3 LeRobot：较直接的 action-expert SFT 备选

官方 π0.5 PyTorch 配置和实现包含 `train_expert_only`、gradient checkpointing；冻结 VLM 后保留 expert 与适配投影学习。默认归一化与镜像/数据 feature 约定必须与导出数据一致，尤其不要把 mean/std 和 quantile stats 混用。[配置源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py)；[模型与冻结实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py)

加载 pretrained 权重和加载整套 policy config/preprocessors 的入口并不等价。若要改 H、camera keys、state/action features，必须检查最终 resolved config；从这里训练后迁往 RLinf，还需要 preprocessing 与固定噪声推理对齐。[LeRobot π0.5 官方指南](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx)

### 1.4 RLinf 已有真机 RLT，不必从仿真架构凭空设计

官方提供 Franka 的 RLT Stage 1/Stage 2，包含冻结特征模型、小 actor/critic、真实环境和人工交互；新的官方路径使用 `openpi_rlinf`。这证明 EnvWorker→真实硬件的映射有现成架构，但 Franka 状态、相机、夹爪和人工介入实现仍要改为本机语义。[RLinf RLT 官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)

| 官方源码 | 读到的具体区别 | 不可照抄的内容 |
| --- | --- | --- |
| [Stage 1 SFT 配置](https://github.com/RLinf/RLinf/blob/main/examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml) | π0.5、SFT task、特定 horizon/相机/token 配置 | 默认精度/batch/冻结策略不是本机低显存配方 |
| [Stage 2 配置](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml) | 单真机环境、actor/rollout 放置、小头训练 | Franka proprio 19D、spacemouse、reset 和设备拓扑 |
| [model_builders.py](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi_rlinf/utils/model_builders.py) | 所读 `_build_sft_model` 直接构造 SFT wrapper，RL builder 才显式讨论 expert-only freeze | 不能仅凭 SFT YAML 的 `train_expert_only=true` 认定冻结生效；须检查完整 factory 和梯度/权重 |

这里最后一项是代码路径审查提示，不是已通过实验确认的上游 bug。

### 1.5 RLT 原始工作与另外一个真机开源参考

原始 RLT 用压缩视觉语言表征和参考动作帮助小策略在线学习，包含表示预训练与后续 RL；论文基座是 π0.6。我们的 π0.5 方案参考方法及公开实现，不标为原论文同模型复现。[原作者研究说明](https://www.pi.website/research/rlt)；[论文](https://arxiv.org/abs/2604.23073)

另有社区实现 `Yyshadow/openpi-RLT`（不是 PI 官方实现），仓库展示 π0.5 + 真机 Ethernet insertion，并拆分模型 serving、在线 learner/replay 和机器人 bridge。这与 charger 插入的工程问题较接近；其 README 的效果是该项目作者报告，未独立复现，设备是 AgileX/ROS 关节链路，不是本机 UR5 TCP。[项目与插接演示说明](https://github.com/Yyshadow/openpi-RLT)

其 runtime 明确处理参考 warmup、局部 critical phase、回合末 replay、人工结果和独立 eval，可作终端状态机/日志设计参考；20 Hz、joint delta、warmup 600/20k updates 等默认值不搬用。已有 RLinf 主线可保留，不为借鉴这些机制额外更换整套训练框架。[在线 runtime 设计](https://github.com/Yyshadow/openpi-RLT/blob/main/rlt_online_rl/README.md)

## 2. 用户仓库：优先取哪几条分支

本轮得到 37 个实际分支（不重复计算 `origin/HEAD` 别名）。完整名和 40 位 SHA 见 [BRANCHES.md](BRANCHES.md)。检查覆盖 tip、目录/关键路径与差异；以下重点分支做了更深源码阅读，其他实验族没有逐个重跑或穷尽历史日志。

| 分支 / commit | 内容与判断 | 用法 |
| --- | --- | --- |
| `codex/rlt-pi0-robotwin` / `2b8199d8` | 早期 RLT 全任务移植，parallel 重建，完整训练/资源报告 | 理解原设计和历史证据，不作为唯一最新实现 |
| `codex/sz-rlt-checkpoint-diagnosis` / `f3ea5f69` | 当前接口移植、checkpoint/契约诊断相关 | 对照 AR 分支变更来源 |
| `codex/sz-rlt-pi0-robotwin-ar` / `d3acd650` | causal AR 重建，动作一次解码、route、续训契约测试 | RLT 核心首选 donor；仍需 π0.5 + UR5 适配 |
| `codex/sz-pi05-robotwin-rl` / `ae7e5da7` | π0.5 仿真 RL/SFT 接入 | 模型配置参考，不搬多卡 batch 和全任务算法 |
| `codex/sz-sidney-pi05-current-rlinf` / `1d015a2a` | LeRobot π0.5 权重转换、固定输入 parity、GRPO 配置 | π0.5 格式/输入对齐 donor；不是 RLT 算法底座 |
| `codex/sz-sidney-pi05-grpo-dvac-adv` / `746f3963` 及 π0.5 BC/DVAC 分支 | 后续仿真算法变体 | 记录存在，首版不纳入 |
| `rlt-teacher-dvac-weighting`、`rlt-dvac-success-episode-bc`、`rlt-dvac-pure-reference-bc`、`sz-rlt-dvac-pure-single-gpu`（均 `codex/` 前缀） | BC 目标/权重与单 GPU 实验变体 | 可后续专项比较，首版不混合 |

### 2.1 RLT 核心移植表（固定 d3acd650）

| 原始文件 | 可复用的逻辑 | UR5 必改/验证 |
| --- | --- | --- |
| [rlt_token_transformer.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/modules/rlt_token_transformer.py) | token encoder；decoder shifted teacher forcing + causal mask；masked reconstruction | 两有效相机/π0.5 state-text 输入后的 prefix 与 valid mask；不能硬抄 768/1024 |
| [openpi_action_model.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/openpi/openpi_action_model.py) | frozen prefix 提取、RLT obs、raw template + decode context、一次 inverse transform | 新 `ur5_tcp7_canonical_v1`；proprio 统一归一化；移除 ALOHA 与 hardcoded 特例 |
| [openpi 模型 factory](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/openpi/__init__.py) | expert-only freeze、token-only freeze、norm stats 从 checkpoint 绑定 | 多处 `strict=False` 必须改成显式键审计；核验 π0.5 base 来源/格式 |
| [rlt_mlp_policy.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py) | actor/critic 使用 z、proprio、reference；固定 std、tanh 动作 | 7D/K 配置与实际动作域；它不是默认 residual-add actor |
| [fsdp_rlt_ac_policy_worker.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py) | twin-Q、BC − Q actor objective、无 entropy alpha 更新 | reward_horizon 当前取张量宽度；中途停止的真实 k/valid mask、warmup/UTD 需处理 |
| [route.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/algorithms/rlt/route.py) | Realworld / FullTask actor/reference 路由 | Realworld `record_transition=actor_switch` 会排除参考 warmup；需解耦 |
| [transition.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/algorithms/rlt/transition.py) | cached current/next feature 与 compact transition；可显式 opt-in | 人工 terminal label、reset 前 final obs、执行源/数量与版本一致 |
| [realworld_env.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/envs/realworld/realworld_env.py) | gym 风格 obs、chunk_step、terminal/reset 包装 | 不是 UR5 驱动；不能 term 后仍执行整段 chunk；人工 reset 要显式等待 |
| [keyboard_listener.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/envs/realworld/common/keyboard/keyboard_listener.py) | 事件与 wrapper 分离的结构 | evdev 物理键盘与用户要求的前台终端不同；用本地 UI + IPC |
| [test_robotwin_rlt_current_port.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/tests/unit_tests/test_robotwin_rlt_current_port.py) | freeze、decode-once、canonical replay、route 真值表、resume contract | 为 UR5 7D、人工状态机、partial chunk、stats mismatch 增加针对性测试 |

注意 `extract_rlt_obs` 的 identity adapter 返回的是 output transform 之后的动作，而 RoboTwin canonical adapter 保留归一化动作与模板再解码。直接换 environment 名称会丢掉这层语义保障。

### 2.2 π0.5 权重转换：可以借鉴，不能当成通用按钮

Sidney 分支的 [lerobot_pi05_to_openpi_rlinf.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591/rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py) 去掉 `model.` 前缀，并按 legacy `OpenPi0ForRLActionPrediction` 校验完整 key/shape。虽然文件名包含 `openpi_rlinf`，实际目标不等于新 vendored 后端。它绑定 π0.5、H=50、Sidney asset、mean/std 处理和相应 feature 假设。

同分支 [sidney_pi05_parity.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591/toolkits/lerobot/sidney_pi05_parity.py) 在不同进程/环境里用固定图像、state、prompt、noise 对比 LeRobot/RLinf 推理。UR5 迁移应沿用这种方法，并覆盖 7D、两有效相机、非训练样本及边界值，而非仅确认文件能加载。

### 2.3 50-task 与 clean-50 的证据区分

`SidneyXie/pi05_robotwin` 模型卡写明从 `lerobot/pi05_base` 微调，数据为 50 个 RoboTwin 任务、27,500 episodes，双臂 14D、三图、30 FPS；报告的评估覆盖其中 32 任务。这与近期 Sidney 分支的加载路径相符，但不是确认旧 home 中哪一个 checkpoint 仍在，也不是 UR5 物理验证。[模型卡与 feature 定义](https://huggingface.co/SidneyXie/pi05_robotwin)

早期 RLT 文档则明确为 `adjust_bottle` 单任务 `clean-50`，使用 π0 SFT 初始化、image-only token、冻结 VLA。AR 后续 artifact 也仍是 π0 任务权重，不能拿 token checkpoint 直接套到 π0.5 UR5。[原始移植计划](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2b8199d8ab2e7b110994fd3234bf7007196c3af9/docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md)

## 3. 本地逐层核验

本表路径相对仓库根；本次未修改其中实现。

| 证据 | 实际行为 | 设计影响 |
| --- | --- | --- |
| [schema.py](../../../src/ur5e_real/data/schema.py) | schema v2；连续化旋转向量；14D 兼容向量；旧第三图复制 wrist | 新 π0.5 数据提取物理 7D，不改历史数据语义 |
| [convert_hdf5.py](../../../src/ur5e_real/data/convert_hdf5.py) | `joint_action/left_arm`、`right_arm` 都填 TCP pose；含 source_run_id | 字段名不等于关节角；保留源数据追溯 |
| [realsense.py](../../../src/ur5e_real/hardware/realsense.py) | `rs.format.bgr8`；两真实视角 | 显式 BGR→RGB，offline/live 共用 |
| [rtde.py](../../../src/ur5e_real/hardware/rtde.py) | 当前采集订阅 timestamp、actual_TCP_pose | 没有关节/力反馈字段；插入不能假设已有接触闭环 |
| [infer_dp.py](../../../src/ur5e_real/adapters/robotwin_dp/infer_dp.py) | DP 模型→TCP chunk→夹爪；同步推理/执行 | 复用执行层，但不能沿用 DP 延迟假设 |
| [chunk.py](../../../src/ur5e_real/control/chunk.py) | 10 Hz 目标转 500 Hz 连续参考；限速；无取消参数 | 加停止/实际 k；请求轨迹、限幅命令、反馈分开记 |
| [gripper_policy.py](../../../src/ur5e_real/control/gripper_policy.py) | 阈值滞回、stable_count=3、2 秒间隔、默认单次开闭周期 | reset 每局；确认 charger 夹爪是否需要不同策略 |
| [terminal.py](../../../src/ur5e_real/collection/terminal.py) | tty/termios/select 轮询，非 TTY 不启用 | 可复用前台 UI，不放进 Ray stdin |
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | 硬件/数据/策略 adapter 分层，手动重播独立 | `Observation/ActionChunk` 在文档中是契约描述，不等于本次已实现新数据类 |

执行层进一步注意：`on_waypoint` 回调在策略 waypoint 边界运行，不是独立高频停止监控；相机读取/串口阻塞都可能影响同一执行时钟。新实现需测回调耗时、任务超期、观测年龄和 watchdog；不能把 dry-run 日志当机器人已经运动的证据。

UR 官方将 servoJ 定义为在线关节位置控制，并说明目标噪声/低更新频率与增益、lookahead 会影响稳定性；它本身不是力控插入接口。RTDE 提供 `actual_TCP_force` 等字段。第二轮补充核验：**collector 未订阅 q/force；执行器 XML 已订阅，但 Python 控制类没有缓存/落盘这些字段**，不能笼统称全仓库未订阅。控制器版本、标定和有效频率仍未现场核验。[servoJ 官方语义](https://www.universal-robots.com/manuals/EN/HTML/SW5_24/Content/prod-scriptmanual/all_scripts/servoj_qavt0-008lookahead_time.htm)；[RTDE 官方字段](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html)

数据盘只读核对的口径是 `/data/robotics/ur5e-real/raw/action/session_*.json` 的 `task/outcome` 字段：cube 18 success/2 aborted，charger 3 success/2 failure，共 25 records。未覆盖逐视频人工质量复核、坏帧、成功定义一致性。失败数据保留给 RL 分析，但不冒充成功 SFT demonstration。

## 4. 显存：公开参考、历史记录与本机推断分开

| 类别 | 数字/条件 | 可得出的结论 |
| --- | --- | --- |
| 本机只读实测 | RTX A6000；49140 MiB total；查询时 623 MiB used；驱动 580.95.05 | 正常可见 GPU；这不是训练峰值 |
| OpenPI 通用说明 | 推理 >8 GB，LoRA >22.5 GB，全参数微调 >70 GB | 只能作公开参考，不能替代指定 π0.5 expert-only 配方实测 |
| RoboTwin π0.5 特定说明 | LoRA >46 GB，full >100 GB；带其配置/batch 条件 | 与 OpenPI 表不矛盾到必须二选一，训练设置不同 |
| 用户旧 RLT Stage 1 报告 | π0、parallel token 重建、micro16/rank、2 ranks，26,447 MiB/card | token-only 可行性的历史线索；不是 AR/π0.5/A6000 的保证 |
| 用户旧 Stage 2 报告 | 两张 A800、8 个仿真环境，GPU 峰值约 19.37/19.56 GiB | 大 VLA 冻结后 RL 可较轻；不要照抄仿真进程与缓存规模 |

数字来源分别为 [OpenPI requirements](https://github.com/Physical-Intelligence/openpi#requirements)、[RoboTwin π0.5 training](https://robotwin-platform.github.io/doc/usage/Pi05.html#5-finetune-model)、[Stage 1 历史记录](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2b8199d8ab2e7b110994fd3234bf7007196c3af9/docs/rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md)、[Stage 2 历史记录](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2b8199d8ab2e7b110994fd3234bf7007196c3af9/docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)。这些报告未在本机复跑。

工程推断：48 GB 优先尝试 expert-only、小 batch、分阶段运行；数据数量影响覆盖、迭代次数和缓存，主要训练显存仍由冻结/可训练参数、梯度、optimizer、activation、图像/token 数和模型副本决定。尤其 RLT encoder/decoder 是 Transformer，不可把 Stage 2 小头参数量用于估算 Stage 1。

第一轮资源实验必须包含一次真实 optimizer step：只看模型 load 或 forward 成功不足以证明 Adam 状态和 backward 能装下。冻结 VLM 还必须核验是否避免不必要的激活图；仅减少 optimizer 参数并不自动最大化显存节省。

## 5. 证据限制与下一步审查

- 官方 `main` 和网页内容会变，本表访问日期为 2026-09-07；本地/user donor 用固定 SHA 可复核。实际实施时还要锁依赖和基础权重 revision/hash。
- 37 分支快照不意味着全部算法等价验证；只抽查了与此次迁移有关的文件/差异，没有宣称全部实验最佳或可复现。
- 当前没有 π0.5 新数据集、训练 checkpoint、转换 parity 报告或真机 RLT 成绩；规划文件本身不算验证产物。
- 旧 `/home/ur5` 目录受访问限制；其权重/cache 当前状态未验证。本轮没有改权限、搬迁旧数据或保存凭据。
- 下一步最值钱的是一批数据的严格加载/冻结/往返检查和 charger 任务定义，而不是再合并更多算法分支。
