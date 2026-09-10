# UR5e：π0.5 → RLT 真机最简闭环规划

**最新入口：[已成功的π0.5方块 → RLT分阶段接入](CUBE_PLAN.md)**（2026-09-10）。
现在规划的是joint-space方块RLT，保留成功的原生JAX基线；下文TCP/charger/统一后端方案仅为历史。
本轮确认：按阶段验收，token先500更新、每100诊断；真机每轮4局串行终端交互，轮末更新和复核。
当前决策、指标、日志与保存约定见[训练阶段讨论稿](TRAINING_STAGES_DISCUSSION.md)；尚未开始RLT训练。
软件已经实施，实际状态与检查见[实施记录](IMPLEMENTATION.md)，新命令见[USAGE](USAGE.md)；以下历史路线不覆盖当前joint版实现。

状态：第一轮讨论稿 v0.1，2026-09-07；保留为历史调研。已调研、未实施。

**当前以 [独立 π0.5 规划](../pi05/README.md) 为准。** 用户在第二轮选定 RoboTwin 原生 π0.5、首版 joint-space 并先补采双记录，只验收方块最小跑通。下文的统一 RLinf 后端、旧 TCP 首版、7D 必须改造及 charger/RLT 决策不再是当前实施要求；charger、RLT、泛化评估均暂缓。

本目录是本任务的持久上下文。后续先读 [决策与续接入口](DECISIONS.md)，再读本文；出处、分支快照及检查边界见 [调研记录](RESEARCH.md)。公开操作接口尚未变化，因此本轮采用中文讨论文档，不替换现有英文 runbook。

## 1. 已确定的目标与结论

- 用户已选：先用 `pick_place_cube` 跑通 π0.5，再做 `charger_plug` 的 RLT。
- 用户已选：首版由人通过终端确认复位、开始、成功/失败、停止；不做连续遥操作接管。
- DP 和手动录制/重播保留，继续作为对照；复用现有真机硬件与 500 Hz 执行层。
- RoboTwin 原生有 `policy/pi05`，但面向 ALOHA 仿真，并非可直接用于本机 TCP 动作的 UR5e 驱动。OpenPI 也有 UR5 示例，不过是关节空间接入提纲，不是本项目的即插即用实现。
- 你的仓库已有值得复用的 π0.5 接入和 RLT 实现；推荐分别取其模型接入经验、动作转换校验、RLT 自回归重建及 replay/续训逻辑，不整合所有实验分支。
- 本机 RTX A6000 48 GB：推理、低 micro-batch 的 action-expert SFT、冻结大模型后的 RLT 是合理候选；尚未实测这些训练的显存峰值。5 条数据不会把模型/梯度/优化器显存变成原来的五十分之一。
- 真机 RLT 不只是替换 EnvWorker：还涉及人工终止与 reward、跨 reset 的 transition 边界、动作实际执行长度、参考策略样本入库、单设备独占和 checkpoint 一致性。

## 2. 本地起点

核验基线：本仓库 `main@aeadb3eb7cada7abaa6509ac2f4a6c719113bd0f`；RoboTwin 锁定 `210720340637cb4619283b295dde4cdd807c9e66`。

| 项目 | 已核验事实 | 对迁移的影响 |
| --- | --- | --- |
| 物理动作 | TCP `[x,y,z,rx,ry,rz]`，米/弧度，旋转向量；夹爪命令状态 0/1 | 不把现有数据误当关节角 |
| 旧 14 维格式 | `[tcp6,0,tcp6,gripper]`，是单臂兼容映射 | π0.5 新适配直接使用物理 7 维，不学习重复的假双臂 |
| 相机 | head + wrist 两台；采集 BGR；旧格式复制 wrist 作为第三视角 | π0.5 统一转 RGB；缺失第三视角使用明确的空图和 mask |
| 时序 | 数据/策略目标 10 Hz，servoJ 500 Hz | 预测 horizon 和实际执行长度必须分开 |
| 夹爪 | 当前模型数据为二值命令状态，不是实测开口宽度 | 连续第二次 close 无法从二值状态学习出来 |
| `pick_place_cube` 原始记录 | 18 success，2 aborted | 可选 5 条成功做最小训练，其余按 episode 留出验证 |
| `charger_plug` 原始记录 | 3 success，2 failure | 不是 5 条成功示教；若按 5-success SFT，至少还需 2 条合格成功记录 |
| 算力 | A6000，49140 MiB 显存；约 123 GiB RAM；根盘约 459 GiB 可用 | 单 GPU 分阶段运行；避免同时常驻多个大模型副本 |

计数仅核对 manifest；尚未逐条视频复核、重新裁剪或确认成功标签。旧 DP 训练快照使用 17 条成功，与当前新增至 18 条不矛盾。

## 3. 推荐代码路线

### 3.1 主线与备选

**优先验证同一 PyTorch 模型后端贯穿 SFT → token → RLT。** 候选底座为你仓库的 `codex/sz-rlt-pi0-robotwin-ar@d3acd650`：沿用其 legacy `model_type: openpi`，新增干净的 `pi05_ur5_tcp7` 配置、数据变换与严格权重检查。它已有 π0.5 模型配置入口、expert-only 冻结入口和成熟一些的 RLT 移植测试，但尚无本项目端到端成功证据。

第一道关卡不是启动正式训练，而是：加载 π0.5 base → 一批 UR5 数据前后向 → 检查冻结参数与权重键 → 保存/重载 → 固定输入输出一致。该候选能否成为正式底座，由这道检查决定。

**备选：官方 LeRobot π0.5 PyTorch 做最快 SFT。** `train_expert_only` 和 gradient checkpointing 有明确实现；随后借鉴你的 Sidney 转换器，将权重与预处理一起接入 RLT。代价是多一个格式转换/数值对齐关卡，不能只转换张量名。

不默认用 RoboTwin 原生 JAX full-finetune 脚本：其 ALOHA 变换、默认训练规模和仿真依赖不适合直接套入本次单卡真机 demo。新 RLinf `openpi_rlinf` 也是可选后续底座，但和你现有 legacy `openpi` 不是同一个模型实现，不能混用配置和 checkpoint。

### 3.2 模型来源

第一轮优先干净 `pi05_base`，用本机任务成功示教微调 action expert 与相应投影；不默认拿 RoboTwin 50 任务 checkpoint 当 UR5 策略。

你的 Sidney 路线与公开 `SidneyXie/pi05_robotwin` 相符：50 个任务、双臂 14 维关节动作、三相机的仿真权重。可以作为以后比较的初始化，但不是物理动作语义相同的理由。另一个名称 `clean_50` 在 RLT 分支里指单任务 50 条示教；二者不要混淆。

### 3.3 架构边界

```text
本仓库：原始记录 → UR5 7D 数据转换 ─────────────→ SFT / token 训练
                   ↓ 同一预处理版本                  ↓ checkpoint
终端前台 UI ←本地 IPC→ UR5 环境桥 ←→ RLinf EnvWorker ←→ RolloutWorker
                       ↓                              ↓ 冻结 π0.5 + token
                  现有硬件/执行器                 replay → 小 actor/critic
                       ↓
                 UR5e + 夹爪 + 相机
```

底层不导入 RoboTwin/SAPIEN。模型运行在隔离的 GPU 环境；硬件进程保留当前依赖，CPU 侧通过本机 IPC 服务观测和动作。首版只允许一个物理环境、一个硬件控制所有者；学习更新优先安排在场景复位时，避免影响动作循环。

## 4. DP → π0.5：必须实现的适配

以下是拟定契约，不是当前已有可调用的新接口。

| 契约项 | 首版建议 | 验收方式 |
| --- | --- | --- |
| state | 连续化 rotvec 的 TCP6 + commanded gripper1 | raw/HDF5/live 三路相同物理值 |
| action | 绝对 TCP 目标6 + 绝对夹爪1；不加 ALOHA joint flip/gripper angular 变换 | normalize → inverse 往返；固定单位/坐标系 |
| 时间对齐 | `obs[t]` 对应下一采样目标 `action[t]=state[t+1]`，H 步不得跨 episode | 与现有 DP shift 规则对照；检查图像/状态时间戳 |
| 图像 | head/wrist，BGR→RGB，统一 resize/pad 到 224；第三视角空图且无效 | 离线与在线同一变换；彩色样图肉眼复核 |
| 语言 | 每任务固定明确 prompt，与 dataset task 映射绑定 | 训练/推理使用同一字符串和 tokenizer |
| 模型维度 | 物理 7D；模型内部按需要 pad 到 32D | 非物理维度不得驱动机器人；明确 padding/loss mask |
| 归一化 | 本任务训练集独立统计，夹爪保持明确开闭约定 | 不复用 ALOHA/DP/旧 UR5 joint stats；不泄漏验证集 |
| 输出长度 | 初始 H=50；cube 先执行 K=6，charger 候选 K=2–5 | K 由真实延迟与任务需要确定，不直接执行全部 H |

为何首版选绝对 TCP：现有数据就是这一语义，最少引入额外编码。π0.5 可以通过自定义变换学习它，但成效需训练验证。若后续需要 delta，另建 encoding 版本：以 chunk 起始观测为共同锚点，只做一次逆变换；不能把相对同一锚点的 H 个目标当增量累加。大角度旋转再考虑严格 SO(3) 相对表示，不悄悄改变已有 rotvec 契约。

特别检查：H=50 在本机 10 Hz 是 5 秒预测，不等于仿真 50 Hz 下的 1 秒。同步推理会使机器人在 chunk 间保持末目标；必须记录 warm/cold latency、p50/p95、图像年龄和实际动作间隔。先测再决定缩短 horizon、执行 K 或改异步预取。

### 4.1 拟新增/修改的位置

| 位置（新路径是建议，尚未创建） | 工作 |
| --- | --- |
| `data/pi05_dataset.py`、`adapters/pi05/transforms.py` | 7D 导出、episode 边界、RGB、prompt、stats、编码版本 |
| `adapters/pi05/client.py`、`infer_pi05.py` | 远程模型调用、结果校验、时间戳、K 步执行；保留 DP 入口 |
| `control/chunk.py` | 取消检查、实际执行计数/时间、限幅结果、超时终止；保持现有重播行为 |
| `control/gripper_policy.py` | episode reset 与任务化策略；核对稳定帧数/2 秒间隔/单周期限制 |
| `adapters/rlinf/env.py`、`operator_bridge.py` | 单环境、人工 reset/reward、终止观测和 IPC；不在 Ray worker 中读 stdin |
| RLinf 派生工作区 | `pi05_ur5_tcp7`、UR5 env 注册、canonical action adapter、route、transition/checkpoint 补丁 |
| `tests/` | 数据往返、冻结、颜色/时序、fake-env 人工流程、短 chunk/终止/replay/续训 |

前三项等本轮讨论收敛后实施；不在当前已工作的硬件环境里原地升级整套 Torch/Transformers/JAX。

## 5. RLT：可以复用什么，真机必须补什么

### 5.1 分清三个训练阶段

1. **任务 SFT**：π0.5 基础模型适配本机任务，冻结 VLM，训练 action expert/相关投影。cube 完成后，charger 也需要自己的任务基线；cube 成功不代表 charger 基线已完成。
2. **RLT token Stage 1**：冻结任务 VLA，训练 token encoder/decoder。优先复用带 causal mask 的 AR 重建；检查真实 token 相比 shuffle/zero token 的重建差异，不能只看 loss 下降。
3. **RLT Stage 2**：冻结 VLA 和 token encoder，仅更新小 actor/critic；解码器不参与在线动作推理。参考策略给出动作 chunk，actor/critic 使用 token、proprio、参考动作等信息。

此处 RLT 指连续隐表征与小 actor/critic，不是离散语言 token PPO。你分支的 actor 输出规范空间中的动作 chunk，不是默认“在原动作上加残差”。论文采用 π0.6；用公开 π0.5 做的是工程适配，不宣称完全复现论文。

### 5.2 首版人工状态机

```text
WAIT_RESET → 人确认场景就绪 → READY → 人开始 → RUNNING
    ↑                                           ↓ 停止/结束/超时
    └──────── COMMIT ← 人确认标签 ← STOP_AND_LABEL
```

拟定终端操作：`r` 确认复位，`s` 开始，空格停止，`y/n` 标注成功/失败，`a` 放弃本局，`q` 退出。按键只是提案，不替换已有采集快捷键；命令需有 `episode_id`、当前状态检查和回执，避免积压按键作用到下一局。

人工摆场、等待标签不计作策略时间步；停止后先保存 reset 前的 final observation，标签确认后提交 terminal transition，再允许下一次 reset。软件停止不替代机器人本体停止机制。

你的旧 keyboard wrapper 用 Linux evdev 读物理键盘，不是终端；Ray 的 worker 也通常没有前台 TTY。复用本仓库 `TerminalKeyPoller` 做独立前台，由 IPC 通知硬件环境和 EnvWorker。

### 5.3 七个不能省略的移植点

1. **参考策略样本入 replay。** 现有 `RealworldRLTRoute` 把“actor 是否接管”和“是否记录 transition”绑定；首版应拆开，参考 warmup 的有效经验也入库。可借鉴 `FullTaskRLTRoute`，但不可照抄仿真 episode 规模。
2. **途中停止立即停止剩余目标。** 现有真实环境模板会循环整段 chunk；本机 `stream_tcp_chunk` 尚无取消参数。需记录实际执行 k、有效 mask，不能把未执行动作当作经验。
3. **三种动作都要留证据。** 记录策略选中动作、解码后请求动作、限幅后发出的命令轨迹，以及实际 TCP 反馈。学习动作语义保持一致；参考阶段不能存未执行的 actor 动作。
4. **人工 reward 与 done 正确对齐。** 成功末步 reward=1，已确认失败=0；任务终止后不 bootstrap。时间限制与基础设施故障分开；未知标签/缺失末观测的数据不能伪装成有效 TD 样本。
5. **终止前后不能串局。** 下一局人摆回起点的图像不是上一局 `next_obs`；人工等待和 reset 过程不进入连续轨迹。
6. **单机调度与资源所有权。** `num_envs=1`，评估与训练不得同时连接机器人；GPU 侧 actor/rollout 尽量共享冻结模型或错峰，不能照抄多卡、多仿真环境配置。
7. **可恢复性。** 权重、optimizer/target、replay 位置、episode/step 计数、RNG、动作编码与 stats hash、token/VLA hash、人工标签一起成套保存；重启先回到等待人工复位，不自动继续旧动作。

### 5.4 动作空间和 TD 的明确约束

RLT actor 常用 `tanh` 输出 [-1,1]，不能直接解释为米/弧度；TCP rotvec 可能超过该数值范围。新 `ur5_tcp7_canonical_v1` 必须定义固定可逆映射，并用同一实现服务 reference、actor、replay 和 inverse decode。建议在 SFT 的 normalized action 外加固定每维尺度 b，使许可的物理范围落入 actor 输出域；b、物理边界、stats 一起冻结。参考动作超出许可域时报告/拒绝，不静默裁剪 BC 目标。仅 q01/q99 归一化也可能超出 [-1,1]，不能据此假设已经兼容。

夹爪闭锁/稳定计数和速度限幅会使“请求动作 ≠ 最后发出的命令”。记录这些执行状态；MVP 拒绝严重投影/异常轨迹用于更新。若改用反算后的 executed action 训练，必须在数据契约里明确并统一 reference/BC/critic，不能随手换 replay 中的 action 字段。

对执行了 k 个策略子步的 chunk，目标应使用有效奖励及真实 k：`Σ(i=0..k-1) γ^i r_i + bootstrap_mask × γ^k Q_target(next)`。gamma 的时间单位写入配置；本机按 0.1 秒子步定义，不能机械沿用 50 Hz 仿真的折扣。部分 chunk 及末步奖励位置需测试。人工 abort 可保存已知有效前缀，但未判定的末尾单独隔离；不把操作中断自动当失败。

## 6. 显存与最快验证策略

| 阶段 | 判断 | 首次验证配置/方法 |
| --- | --- | --- |
| π0.5 推理 | 48 GB 有充足的尝试空间 | 单模型、两有效相机；先固定输入推理与计时 |
| action-expert SFT | 有希望；未在本机测量 | micro-batch 1→2→4，梯度累积；检查 VLM 确实冻结；按后端支持启用 checkpointing |
| 全参数 SFT | 不作为单卡首版 | 官方 OpenPI 通用参考 >70 GB；RoboTwin 特定说明更高 |
| token Stage 1 | 不等于训练一个很小的 MLP | encoder/decoder 本身有多层 Transformer；单独测前后向/optimizer step 峰值 |
| RLT Stage 2 | 小头训练有利于单卡；仍要装得下冻结特征模型 | 优先一份冻结 VLA，回合之间更新，控制 replay/cache 与进程副本 |

公开 LoRA/全参数数字依赖实现、batch 和精度，不等于本项目 expert-only 保证。你的早期 RLT 仿真日志给出 26,447 MiB/卡的 token Stage 1 峰值，但那是另一设备/配置/重建版本，只作可行性线索。

每次资源检查同时保存：软件锁版本、精度、可训练参数数、optimizer 参数数、micro/global batch、H/K、相机/token 数、allocated/reserved 峰值、进程级 nvidia-smi 峰值、step/p95 inference 时间。5 条数据首先检验管线能否拟合；不把过拟合成功当作场景泛化。

## 7. 里程碑与验收

| 阶段 | 交付物 | 通过条件 | 暂不做 |
| --- | --- | --- | --- |
| P0 接口冻结 | schema、选定 5 条 cube manifest、版本锁、严格加载报告 | RGB/7D/时间 shift/归一化往返一致，参数冻结可证 | 真机动作 |
| P1 cube SFT | checkpoint + stats + 数据清单 + 显存/损失记录 | 一批训练/保存重载通过；小样本可拟合；独立 episode 离线检查 | 声称泛化成功 |
| P2 cube 真机 demo | shadow 延迟日志、受控执行记录、逐局结果 | 相同受控摆场可完成抓取放置；所有尝试均留档 | 自动场景复位/多任务 |
| P3 charger 基线 | 明确成功标准、近插入初态、至少 5 条合格成功候选、任务 SFT | 参考策略与任务几何/执行能力匹配；能够采到有学习价值的回合 | 无约束接触探索 |
| P4 token Stage 1 | AR token 权重及 true/shuffle/zero 对照 | VLA 未改变、无重建信息泄漏、token 依赖可观察、重载一致 | DVAC/多种 RL 算法混合 |
| P5 fake-env RLT | 人工事件回放、replay 样例、短 chunk/终止/续训测试 | 不串局、不学未执行动作，warmup 入库，优化器确实更新小头 | 真机学习 |
| P6 charger RLT MVP | 单机终端闭环、少量真实回合和更新、可恢复 checkpoint | reference→actor 路由、人工标签、有效 TD、停止/复位全闭环 | 承诺成功率一定提升 |
| P7 效果验证 | 冻结参考策略 vs 冻结 RLT 的独立评估 | 同摆场分布/预算，报告全部尝试和干预，不边测边训练 | 把工程跑通当算法优越性 |

P2 可先安排 5–10 次受控尝试，P6 可用 10–20 局作为工程集成预算提案；这不是学习收敛保证。正式比较的回合预算在看到 reference 成功率、人工复位耗时与数据质量后决定。比起直接大规模 RL，先让任务基线在近目标初态下提供有效经验更可能尽快得到 demo。

## 8. charger 插入任务还缺哪些事实

- **成功判定**：插到约定深度、视觉认为到位，还是电气接触？首版由人确认，但标准必须统一。若涉及供电，用适合该实验的连接状态和夹具方案，不默认用 RL 探索带电插接。
- **初始范围**：建议首版已夹住插头、从插口附近开始；完整抓取→搬运→插入另列阶段。这是工程建议，尚未获用户确认。
- **接触执行能力**：当前采集 RTDE 仅 TCP pose/time，不含力反馈；现有 pose servo 不自动等同于柔顺插入。需要确认机械夹具/容差、允许接触与停止条件，必要时补反馈或改变执行策略。
- **夹爪语义**：若 charger 必须二次收紧、多次开闭，现有二值 + 单周期状态机不够；可先通过固定预抓取避开，也可另做事件动作表示，不能由模型调参解决。
- **数据与资产**：核验已有 3 条成功 charger 质量；旧 `/home/ur5` 模型/cache 当前未获可读访问，不能声称已经找到可直接复用的权重。

这些 gap 不阻塞 P0/P1 的离线准备；会影响 P3/P6 的实施范围。下一轮讨论优先收敛 charger 的成功标准与初始状态，再决定是否需要局部阶段切换；不必一开始引入连续人工接管。
