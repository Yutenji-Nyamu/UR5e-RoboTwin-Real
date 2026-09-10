# 已成功的 π0.5 方块策略 → RLT 分阶段接入

**推荐主线：保留成功的原生 JAX π0.5，缓存其冻结特征训练 AR token，再接 PyTorch 小 actor/critic，最后做终端交互的单真机在线 RL。**
已有5条示教和1000步checkpoint不重做；先实现最小闭环，不以泛化评估或 charger 阻塞。
2026-09-10已按本方案完成本地算法移植与真机交互接口，见[实施记录](IMPLEMENTATION.md)和[操作说明](USAGE.md)。
尚未正式训练token/A/C、运行RLT真机回合或验证收益；本地coordinator不等于完整分布式RLinf runner。

训练计数、论文/donor预热差异、验收指标及逐更新保存的讨论，见
[训练阶段与交互讨论稿](TRAINING_STAGES_DISCUSSION.md)。2026-09-10已确认：分阶段验收，token首段500更新、每100诊断；
真机每轮4局串行交互，助手依据日志判断下一步。旧的分阶段固定局数方案取消。
首版据此按4局内固定checkpoint、轮末集中更新和复核组织；其余训练超参数仍是候选，本轮尚未开训。

## 当前起点与方案边界

- `pick_place_cube` π0.5 真机最小demo已由现场操作者确认成功；运行参数/两份执行日志见
  [成功记录](../pi05/PHYSICAL_RESULT_20260910.md)。SFT为原生JAX/Flax，不能当作PyTorch state_dict。
- 采用 `joint5_sft_20260909_01:1000`，5条v3示教、414训练transitions；q6+夹爪，
  双有效相机，H50/K20/10Hz，joint servoJ500Hz。旧TCP DP和短命令作为回退基线保留。
- 人通过前台终端确认场景复位、开始、成功/失败/中止；不做连续遥操作接管。
  当前范围是**完整抓放方块**，不是预抓取charger，也不引入DVAC/GRPO/PPO/IQL变体。
- 本地审查基线`5342f1f`，上游RoboTwin固定`210720340637cb4619283b295dde4cdd807c9e66`。
  2026-09-10刷新用户仓库得到45个分支（不含origin/HEAD），见[新源码快照](SOURCE_SNAPSHOT_20260910.json)；旧[37分支表](BRANCHES.md)保留为历史快照。

RLT在这里指冻结大模型特征后的轻量off-policy actor/critic训练，不是“RL直接改写π0.5全部权重”。
RLinf有Franka真机示例可借鉴，但其设备和模型后端不等于本机。[^1]

## 源码取舍：固定版本，不合并全部分支

| 来源 | 已核验内容 | 本次用途 |
| --- | --- | --- |
| `codex/sz-rlt-pi0-robotwin-ar` @ `d3acd650869d376c14c1406d90865ef90d680432` | token-only AR重建、canonical动作、actor/critic、route/replay/续训契约 | RLT主要donor；此次远端tip未变 |
| `codex/rlt-pi0-robotwin` @ `2b8199d8ab2e7b110994fd3234bf7007196c3af9` | 早期parallel重建与分阶段实验记录 | 历史比较，不把旧吞吐/成功率当本机结果 |
| `codex/sz-rlt-checkpoint-diagnosis` @ `f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4` | checkpoint/接口诊断路线 | 查契约沿革，不先切换执行后端 |
| `codex/sz-sidney-pi05-current-rlinf` @ `1d015a2aa03ec8132d8207ba47a2be3dbe1d9591` | LeRobot→legacy OpenPI PyTorch的key/shape检查、固定输入/噪声parity | 一致性检查方法；**不是当前JAX checkpoint的一键转换器** |
| `codex/sz-pi05-robotwin-rl` @ `ae7e5da72acf4a47a54cecff4f802ebc174b397a` | π0.5仿真配置路线 | 辅助参考，不照搬50任务、多环境、batch和动作语义 |

新增分支主要是09-08/09的π0.5 online-BC/DVAC/PRISM/IQL等实验；本次刷新了分支清单，
没有重跑所有实验，也不宣称AR分支效果最佳。主要donor的方法/接口适合此次范围，才选择它。[^2][^3]

AR donor具体可借：`rlt_token_transformer.py`的encoder、shifted teacher forcing/causal mask；
`rlt_mlp_policy.py`的`z_rl/proprio/ref_chunk`输入；`fsdp_rlt_ac_policy_worker.py`的双Q、
`BC − Q` actor目标、无entropy/alpha更新；`route.py`、`transition.py`和resume契约。
保留来源SHA和Apache-2.0版权/许可，不复制旧机器路径、数据或实验权重。[^2][^4][^5]

## 后端衔接：无需先转换整个 VLA

| 路线 | 优点 | 代价/结论 |
| --- | --- | --- |
| **JAX冻结VLA + PyTorch RLT组件** | 直接复用成功checkpoint和native变换；token网络是独立张量模块，可吃缓存特征 | 新增特征/RPC桥和RLinf feature adapter；推荐先验证 |
| 全量转为donor PyTorch VLA | 更接近其既有rollout factory | 当前Sidney转换器不是JAX入口；须另做参数转换与整链parity，作为桥接失败后的备选 |
| RLT全部重写JAX | 单模型环境 | 重写token和actor/critic/optimizer，复用donor代码较少；首版不选 |

推荐路线是基于已读源码的**工程推断，尚未做跨环境原型**：native `sample_actions()`预填充
PaliGemma时已有最终prefix hidden states/KV cache，但目前丢弃hidden states；本地WebSocket也只返回动作。
因此需要新增明确的`features_and_reference`能力，不是简单给当前RPC增加一个配置名。[^6]

```text
硬件环境：相机 + q/TCP + 夹爪状态机 + 唯一运动控制权
                  │ 带时间/请求ID的原始观测
                  ▼
原生JAX π0.5：冻结checkpoint → prefix/mask + reference H50
                  │ 本机IPC，保留obs/contract/checkpoint身份
                  ▼
独立PyTorch/RLinf：冻结token encoder → z + proprio + reference K20
                  │ reference或actor动作（每轮4局锁定策略版本）
                  ▼
UR5 adapter：一次反变换 → q6/夹爪 → 500Hz joint执行
                  │ 实际执行报告 + 末态 + 终端标签
                  └────────────→ 逐局持久化 → replay → 每4局轮末更新/复核
```

VLA冻结、图像增强关闭时，Stage1可一次缓存414份prefix/mask，再独立训练token；token此时仍参与训练。
这与在线重新提取相同确定性特征有等价条件，必须实测缓存/live对齐，而非默认文件格式相同就等价。
第一步可独立提取prefix以做parity；正式在线应复用一次prefill，避免每次重复大模型前向。
JAX和PyTorch进程独立，不能直接跨进程传GPU指针；先用显式CPU张量/类型协议，测拷贝延迟后再优化。

## 分阶段交付与验收

| 阶段 | 训练/数据行为 | 交付与通过条件 |
| --- | --- | --- |
| S0：冻结成功基线（已具备） | 保留现有SFT和5条数据，不继续训VLA | 固定数据/norm/checkpoint/代码身份；现有两条推理命令不变 |
| S1a：特征桥（先开发） | 无训练、无机械臂；从现有数据抽样 | 同图/state/prompt/noise、10采样步下参考动作与旧服务一致；输出prefix/mask、normalized state、raw reference、decode context |
| S1b：AR token训练 | VLA完全冻结，只训token encoder+decoder；使用5条数据缓存 | 一步更新/保存/重载先通；再拟合；检查冻结哈希、masked reconstruction、zero/shuffle token消融，不能只看teacher-forcing loss |
| S2a：小头初始化 | 冻结VLA+token；先离线BC模仿reference，独立初始化双Q；随后采reference每轮4局，进行Q与强BC actor预热 | actor与实际部署的tanh/采样路径一致；动作域往返和夹爪通过；reference回合确实入replay；由指标判定是否进入actor验证 |
| S2b：在线RLT | 单机械臂，逐局落盘；每轮4局固定checkpoint，轮末更新小头并复核，下一轮同步选定版本 | 终端reset/start/label/abort闭环；支持partial chunk、超时、崩溃恢复；4局是复核节奏，不是训练完成门槛 |
| S3：冻结比较 | learner关闭，actor确定性eval，不加入探索噪声 | 同一摆场分别跑reference和actor并人工标注；先报告闭环成功，是否优于baseline另作结论 |

S2a是Stage2内部的预热，不是重新训练π0.5。donor以`rlt_train_vla=false/rlt_alpha=0`
实现token-only Stage1；Stage2固定std actor+BC−Q，不是标准最大熵SAC，也不是默认“reference+residual”相加。
不能把SFT warmup、token训练步数、replay warmup、actor/critic更新步数统称为“几阶段训练”。[^2][^4][^5]

### 五条数据分别能做什么

- **S1**可直接复用；没有泛化数据要求。缓存应保持episode/run_id/时间，不能把末尾H50 padding当新增观测。
- **actor预训练**优先模仿成功VLA给同一观测的reference；现有SFT标签是实测`q[t+1]`，
  不是历史下发命令，不能伪造成完整在线replay的执行动作。
- **critic预热/在线RL**需要动作来源、真实执行长度、next_obs和结果标签；现有推理日志只有汇总，
  不足以重建这些transitions。后续由新开发的reference rollout采集器生成，**本轮不要求补采**。
- 414个10Hz训练样本不等于414个K20决策transition；现有两次执行各5/4个chunk。
  所以仿真`warmup_min_size=500`可能意味着很多真机回合，不能因“数据5条”直接照抄。
- 全成功数据足以验BC/流水线，不证明Q已学会分辨失败；后续自然失败/timeout要真实标注保存，
  不造失败标签，也不为凑正负样本强行制造碰撞。

## 真机适配必须补的契约

### 1. 模型空间、物理空间、夹爪分别建模

成功基线依旧是`[q6,0,q6,g]`、native pad32、joint delta相对当前观测q、quantile normalization、
`adapt_to_pi=False`；模型输出反变换后取列0..5与13执行。RLT小头建议只用**7D规范动作**，
不把重复两臂或padding列当可探索自由度。reference的7D选择、填回native模板、反归一化和恢复
绝对q只能发生一次；规范7D是新adapter契约，不改旧checkpoint。[^3][^6]

donor tanh动作有界，但native quantile坐标不保证落在[-1,1]。S1a须统计**这批实际reference**的范围，
明确小头归一化比例/边界版本；不能静默clip动作后仍用未clip的坐标训练Q，或对已绝对q再加一次当前q。
保持硬件joint/TCP包络、限速、取消和跟踪停止，不用模型归一化范围替代硬件限制。

夹爪目前是命令状态，不是实测宽度；去抖计数、当前开/关、距上次命令时间、cycle/phase会影响下一次动作。
这些执行状态应作为env状态/必要的policy辅助观测或重置契约，并记录到replay；
仅有q6+g不能完整解释夹爪去抖后的状态转移。首版保持一次close→open；是否支持失败后重抓另行决定。

### 2. chunk与时间：K20保持，实际长度不能写死20

已有成功执行末段分别只执行18和10点，这不是理论边角情况。
`stream_joint_chunk`已有高频`cancelled`钩子和统计，**不必重写500Hz执行器**；缺的是
终端取消接线、异常/partial报告持久化与RL transition契约。[本地执行器](../../../src/ur5e_real/control/joint.py)

每次记录`obs_id`、生成/执行时间、H、requested_K、executed_k、requested action、实际下发/限速情况、
measured q/TCP、gripper事件、末态、policy/version/seed、停止原因。存盘区分“命令”和“跟踪反馈”。
终止发生后不再执行该chunk剩余点；donor旧`realworld_env.chunk_step`会循环完整chunk，不能照搬。[^7]

建议先沿用donor的**10Hz动作步折扣**而非每chunk固定折扣：
`R = Σ(i=0..k-1) γ^i r_i`，bootstrap系数为`γ^k`；初值γ=0.99则完整20点约0.818，
不是0.99。`k`用实执行长度；终端人工等待不累计reward/机器人步数。
记录wall-time与retiming，首版不引入连续时间折扣；若后续需要改变时间单位，另立版本。

donor当前按reward张量宽度求horizon，且BC对整chunk取均值，需要补valid mask。
对Q的宏动作定义也必须统一：保留requested完整宏动作和executed前缀两份；
选用执行前缀训练时，padding mask必须进入Q/损失契约，不能把填零当真实命令。
测试至少覆盖完整20、提前18/10、零动作取消及retime；critic输入和actor目标不得使用互不一致的动作定义。[^5]

### 3. 人工交互不是Ray worker中的input()

建议前台终端独占按键，以带episode_id/序号和ack的本机IPC通知env；禁止把重复按键重放到下一回合。
env拥有唯一设备锁，模型/RLinf进程均不能另建运动连接。

```text
WAIT_RESET → PREPARE/VERIFY → WAIT_START → RUNNING → STOP/FINAL_OBS
    ↑                                                    ↓
    └──────── COMMIT_EPISODE ← LABEL(s/f/a) ←─────────────┘
```

复位时可在确认后回原位/开爪或进入显式人工拖动；不把人碰触设备时的观测当RL样本。
退出RUNNING先停止流并保存final_obs，再等人工标签，不让人等待时触发2秒command watchdog，
也不为适应人的速度而放宽运行中的1.5秒RPC/0.5秒状态期限。
终端关闭、急停、相机失联、控制异常均进入abort路径；先落盘已发生事件，下一次启动不自动运动。

建议结果：成功/任务失败=`terminated`；纯步数上限=`truncated`，只有真实可用的reset前末态才允许bootstrap；
硬件异常/人为中止=`aborted`单独标记，首版不作普通负例进入Q，保留已确认完整片段供审计。
人工label可回写episode结果，不覆盖原事件；禁止用下一回合reset图像充当上一回合next_obs。

### 4. 路由、日志、恢复和资源

- reference warmup同样入库；donor Realworld route把`record_transition`绑定actor_switch，会丢reference样本，
  需要解耦。cube首版按整回合reference/actor选择，不增加插入阶段的键盘接管门。[^4]
- 原始episode用增量日志+完成标记，未完成标注保持pending；图像/状态/命令/特征按时间关联。
  当前短命令的`execution.json`在正常结束才写，不能当成RL replay。先实现记录器，再在线更新。
- learner在每4局的轮次之间集中运行，保存actor、双Q/target、optimizer、replay游标、更新计数、随机状态；
  feature/token/norm版本变化不得混进同一buffer。恢复后保持WAIT_RESET，不自动继续下发旧chunk。
- 同一A6000上保留一个JAX VLA副本、一个冻结token encoder，小头learner按需运行。
  token decoder仅Stage1训练使用。使用独立`.venv/rlt`锁环境，不在已成功的两套环境上叠装RLinf。
  首版若RLinf factory强绑定PyTorch VLA，新增feature provider，不能假装只换EnvWorker即可接入。
- 重用训练资源监控习惯：2秒采样、30秒摘要；分别记录JAX/PyTorch显存、主机可用内存、
  RPC/提特征耗时、观测年龄、执行超期、episode数和真实更新次数。不自动开多环境占用同一机械臂。
- 日志按“episode原始记录 → round汇总 → 逐更新指标/恢复点 → MD决策摘要”维护；
  每轮写明阶段、输入/输出checkpoint、数据范围、人工结果、指标和继续/切阶段理由。
  具体字段与待实现落点见[每轮交互、日志与判定](TRAINING_STAGES_DISCUSSION.md#每轮交互日志与判定)。

## 参数：哪些沿用，哪些仅是试跑建议

| 参数 | 初步方案 / 来源 | 开训前验证 |
| --- | --- | --- |
| 基线H/K/频率/采样步数 | 50/20/10Hz/10；当前成功基线与用户K20决定 | 新reference路径固定噪声parity；不照抄donor K10/4 flow steps |
| Stage1冻结 | VLA全冻、只训AR token，`rlt_alpha=0`；donor token-only配方 | optimizer只有token参数，VLA哈希不变 |
| token维度/层数 | input/z=2048、encoder/decoder各2层、8头、mlp_ratio4；donor初值 | 实际prefix宽度/mask；不硬猜π0与π0.5 prefix长度相同 |
| token范围 | 先全prefix计算，再取image位置和真实mask；donor image_only=true | 缺失左腕不能当有效图；π0.5 state在prompt中，必要时对比全prefix，不先换成1024假定 |
| token batch/步数 | 已确认首段500步、每100诊断；每100保存及缓存后batch1→4→8每档3步仍是实现候选 | 缓存live对齐、loss/消融决定是否续训；不把batch候选记为已确认 |
| token optimizer | AdamW，lr2.5e-5、warmup100；donor参考 | 几步短测warmup设0；不是继续SFT optimizer |
| Stage2预热 | actor先离线BC；reference每轮4局；轮末Q与强BC actor预热，按指标决定续训或再采一轮 | 不照搬500transition/5000update，不预设20–40个chunk硬门槛；更新量与数据复用率一起报告 |
| Stage2更新 | 初始batch32，lr1e-4、tau0.005、双Q；γ0.99/动作步 | 每新transition先约1–4次梯度更新作为工程起点，记录Q/BC和actor偏离；不视作调优结论 |
| 探索/BC权重 | 首先无探索reference和确定性BC小头；再讨论小幅joint探索 | donor fixed_std0.002在tanh前，不是0.002rad；不同关节缩放/夹爪需分开核验 |
| 保存/评估 | 每局保存episode；每轮4局后复核；拟逐learner更新保存小头恢复点，阶段末另存不可变模型；eval无噪声 | 轮内固定checkpoint，下一轮装载选定版本；不把4局结果等同于收敛或泛化结论 |

**显存证据：**本次仅用已审阅的donor独立token模块在PyTorch **meta设备**做参数计数，未初始化CUDA、未训练：
默认token encoder+decoder为745,715,712参数，encoder为370,761,728参数。
按FP32权重+梯度+Adam两状态计算约11.11GiB，另加激活/临时缓冲；不是“小MLP几乎不占显存”。
Stage1先缓存再卸载VLA有利于隔离峰值；Stage2只带冻结encoder及小头，但仍有JAX VLA常驻。
不能把此前SFT约17GiB的峰值当RLT实测显存。[^2]

缓存容量可按`414 × 实际prefix长度 × 2048 × 每元素字节数`计算，例如768位置/bf16约1.21GiB，
这只是估算，不宣称已测得768个有效token。JAX不预分配全部显存，PyTorch模型副本数量和特征拷贝
仍需S1a短测；不先承诺吞吐倍数或完整在线训练时长。

## 需要讨论的少数决策

| 问题 | 推荐默认 | 最晚决定点 |
| --- | --- | --- |
| **cube成功标准** | 方块明确被提起、随后放回桌面、完全脱离夹爪并稳定至少1秒；是否必须落入指定区域待确认 | 真机reward采集前；不等同于“发出open” |
| **先验哪种结果** | 首轮只验RLT全阶段闭环，同一现有摆场；不声称已优于π0.5 | 训练预算/评估前；若要提升成功率，另设计baseline有失败的合法初态集合 |
| **一次尝试还是允许重抓** | 保持当前一次close→open，随后停止并标注；失败重开一局 | env和夹爪episode语义锁定前 |
| **在线回合/探索预算** | 已确认每轮4局串行交互、轮末由助手复核；不预设分阶段总局数；joint与gripper探索设置仍分别确定 | 4局内固定配置，跨轮修改记日志；探索设置在S2b首次执行前确定 |

后端桥、缓存格式、action-domain一致性、partial-chunk损失和显存属于**开发验收事项**，
不要求操作者凭空选择实现细节。可直接开始的下一轮范围是S1a及episode记录/假env状态机；
成功标准和探索预算不阻塞纯离线开发，但不能被规划文档默认为已经批准的在线训练授权。
实现进度已超出初始S1a建议范围，当前软件与后续待验收项以[实施记录](IMPLEMENTATION.md)为准。

## 建议代码落点与后续记录

保持[架构边界](../../ARCHITECTURE.md)：模型/训练归RoboTwin与RLinf，设备与执行归本仓库。
以下保留原规划落点供对照；实际文件和可执行命令见[实施记录](IMPLEMENTATION.md)与[USAGE](USAGE.md)：

| 部分 | 拟落点 | 最小回归 |
| --- | --- | --- |
| JAX prefix/reference provider | `adapters/robotwin_pi05/rlt_features.py` | reference动作parity、图像/state/prompt/mask身份、缓存/live一致 |
| donor锁/隔离环境/token训练 | `integrations/rlt/` | 来源许可证、锁文件、meta/一步训练/重载/冻结/消融 |
| RLinf桥与规范动作 | `adapters/rlinf_rlt/` | 7D↔14D/32D、decode-once、范围、policy版本 |
| 单设备owner/终端协议/日志 | `rl/episode.py`、`rl/terminal.py` | reset前末态、重复按键、取消、partial执行、异常持久化 |
| learner/replay | 优先donorworker/route/transition的窄适配 | reference入库、mask/γ^k、终止/截断、恢复不自动运动 |

每阶段追加：commit/环境、数据/模型hash、配置来源、短测数值、产物路径、未通过项；
用本文件作为RLT续接入口，旧README/DECISIONS中的TCP和charger路线只作历史证据。

## Sources

[^1]: [RLinf官方RLT说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)，访问2026-09-10；提供Franka/ManiSkill两阶段示例，其当前`openpi_rlinf`为vendored PyTorch后端，不是本机JAX模型。
[^2]: [固定AR token模块](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/modules/rlt_token_transformer.py)；[token-only Stage1配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml)。
[^3]: [donor特征/一次decode接口](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/openpi/openpi_action_model.py)；[Sidney转换器](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591/rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py)；[固定输入parity](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591/toolkits/lerobot/sidney_pi05_parity.py)。
[^4]: [MLP策略](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py)；[路由](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/algorithms/rlt/route.py)。
[^5]: [RLT actor/critic损失](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)；[Stage2配方](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current.yaml)；[transition](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/algorithms/rlt/transition.py)。
[^6]: [锁定RoboTwin原生Pi0/π0.5模型](https://github.com/RoboTwin-Platform/RoboTwin/blob/210720340637cb4619283b295dde4cdd807c9e66/policy/pi05/src/openpi/models/pi0.py)；本地[动作契约](../../../src/ur5e_real/adapters/robotwin_pi05/contract.py)、[服务](../../../src/ur5e_real/adapters/robotwin_pi05/serve.py)、[推理](../../../src/ur5e_real/adapters/robotwin_pi05/infer.py)。
[^7]: [donor真实环境chunk循环](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/envs/realworld/realworld_env.py)；本地[夹爪状态机](../../../src/ur5e_real/control/gripper_policy.py)和[取消/执行报告](../../../src/ur5e_real/control/joint.py)。
