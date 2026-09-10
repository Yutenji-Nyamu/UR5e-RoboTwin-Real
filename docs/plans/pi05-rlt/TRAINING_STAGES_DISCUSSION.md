# RLT 训练阶段、预算与终端交互

RLT 需要分开管理**数据量、梯度更新量、谁在执行机器人、阶段验收指标**。现场试跑的回合数，只覆盖第一项和部分执行验证，不能代表 token、actor、critic 已完成训练。

本稿区分论文方法、固定 donor 源码、已确认的组织方式和候选参数；正式运行结果单独记录。
保留已成功的 JAX π0.5，不重做示教或 VLA SFT。架构见 [方块接入规划](CUBE_PLAN.md)，实际代码见[实施记录](IMPLEMENTATION.md)。

## 最新决策与续接状态（2026-09-10）

- **按阶段推进、按各阶段指标验收**；不是达到固定回合数就自动宣告训练完成。阶段结束保存选定 checkpoint 和判定依据。
- **Token 首段预算 500 更新，每 100 更新诊断**，依据曲线与 token 消融决定是否延长；这项预算已确认。
- **真机 rollout 每轮 4 局，单机械臂串行执行、逐局终端交互**。4 是 episode 数，不是 optimizer batch、动作 chunk 数或并行环境数。
- 取消此前按 reference、actor 验证、在线阶段分别分配固定局数的方案；不再作为配置默认值或阶段门槛。
- 操作者负责场景复位、开始、成功/失败/中止标注；助手在正式运行时读取日志，决定继续离线更新、再采一轮、调整参数或进入下一阶段，并记录理由。
- **当前已实现并离线/假环境检查；尚未正式训练RLT或执行真机rollout。**既有π0.5成功与本次软件测试不计入RLT学习效果。

本轮据此采用的实施默认：4 局内固定策略来源、checkpoint 和探索设置；每局完成立即落盘，整轮结束再集中训练、复核和保存，下一轮才装载选定版本。提前结束时保留已发生的尝试和不满 4 局的进度，不隐去失败或为凑成功数自动补局。训练超参数和成功区域/重抓语义仍按各自的待定状态处理，不能因确认了“4 局一轮”而默认为全部参数已确认。

“分阶段”不意味着 actor 与 critic 从此完全分离：token 和 BC 初始化先离线验收；随后使用 reference 数据预热 Q/actor，在线阶段仍联合更新 A/C。首版顺序为：

`特征桥及记录器就绪 → token 离线训练 → actor BC 初始化 → reference 4 局/轮及 Q/A 预热 → actor 验证 4 局/轮 → 在线采集 4 局/轮与 A/C 更新`

这是操作顺序，不要求每经过一个阶段都新增 4 局；token 和 BC 初始化直接使用现有数据，无需真机配合。

## 论文与 donor 的阶段

原论文先适配 VLA 并训练表示，再冻结 VLA/token，进行带参考动作约束的 actor–critic 学习。附录给出的 VLA 适配与 token 训练预算为 2,000–10,000 梯度步；在线部分采用双 Q、critic:actor 更新比 2:1、update-to-data ratio 5。算法没有规定“Q loss 达到阈值后才开始训练 actor”；也没有公布通用的 token/Q/actor 收敛阈值。附录中“50 次基础策略 rollout 后用 Cal-QL 预训 critic”属于对比方法 PLD，不是 RLT 的必需阶段。[^1]

固定 donor 为 `codex/sz-rlt-pi0-robotwin-ar`，commit `d3acd650869d376c14c1406d90865ef90d680432`。以下是已提交配置及其代码含义，不是对某次实验实际完成量的声明。

Stage1 的 `robotwin_rlt_stage1_sft_openpi_current_ar.yaml` 配置：只训练 AR token 模块，VLA 冻结；2,000 步、global batch 32，2,000 步保存一次，关闭周期验证。它按步数结束，没有内置收敛验收。缓存 prefix 只要求产生这些特征的 VLA 固定且预处理一致，**不要求 token 冻结；Stage1 的 token 正是待训练对象**。[^2]

Stage2 有基础方法配置和正式运行覆盖配置，不能混用两个文件的数字：[^3]

| 项目 | 基础方法配置 | `current_8env250` 覆盖后 |
| --- | ---: | ---: |
| 开始预热所需 replay 行数 | 500 | 10,000 |
| 允许 actor 接管前的 learner 更新数 | 5,000 | 30,000 |
| actor 损失权重保持预热值的更新数 | 5,000 | 20,000 |
| 随后的 BC/Q 权重过渡长度 | 10,000 | 50,000 |
| 每新增一条 transition 的目标 learner 更新数 | 5 | 5 |
| 一次 `run_training()` 最多更新 | 400 | 1,600 |
| runner 循环预算 | 0，基础文件不能直接开跑 | 250 |
| checkpoint 周期，单位为 runner 循环 | 10 | 25 |

这里 replay 门槛检查的是各 learner rank 的最小本地 replay 大小；不是把多个 rank 的样本数直接求和后过门槛。基准动作是仿真 ALOHA 14D、K10，并非 UR5 的 7D/K20。

每个 learner 更新都会更新双 Q，actor 每两次更新一次。预热期间 actor 已经训练，只是 rollout route 仍选择 reference。基础配置下，达到接管计数时大约已做 5,000 次双 Q 更新、2,500 次 actor 更新。[^4]

actor 的目标为：

`L_actor = w_BC × BC_error − w_Q × Q_actor`

预热权重是 `w_BC=7.0, w_Q=0.05`，随后过渡到 `2.5, 0.45`。因此 donor 的预热不是 BC-only，更不是只训练 critic。正式覆盖配置中，20,000 次 learner 更新后权重已开始过渡，30,000 次才允许 actor 执行；接管时权重约为 `6.1, 0.13`，还没有走完过渡。代码的 `ready_for_online` 只比较计数，不检查拟合质量。[^4][^5]

“actor 接受梯度”“actor 开始更重视 Q”“actor 接管机器人”是三个不同事件。eval 路由还会直接选 student，不受训练模式的预热 gate 约束；阶段评估应显式选模型版本，不能把 eval 当 reference 预热执行。[^5]

## 预算单位

UR5 当前规划沿用 H50/K20、10 Hz 目标点、500 Hz 底层插值执行。以下计数应分别记录：

| 计数 | 含义 |
| --- | --- |
| episode | 一次完整抓放及其人工结果 |
| executed action point | 真正执行的 10 Hz 目标点；500 Hz 插值点不是新 RL 样本 |
| transition | 本方案首版一个 chunk 决策及其真实后继观测；末段长度可能小于 20 |
| critic update | 双 Q 的一次 minibatch 优化，不是一次机械臂动作 |
| actor update | actor 的一次 minibatch 优化；2:1 调度下数量约为 critic 的一半 |
| runner cycle | 一轮调度，可能包含多次 rollout 和数百次梯度更新 |

已有 414 个 SFT 样本不能直接记作 414 个在线 chunk transitions。其动作标签和采样边界，也不能替代新 rollout 中实际下发的动作与后继状态。[^6]

样本复用强度可用 `更新数 × batch / replay 样本数` 粗看。例如 30 条 transition、batch32、300 次 Q 更新，相当于平均每条被抽到约 320 次。训练步数多说明复用多，不说明获得了同等数量的新经验。这不是泛化要求，而是解释 loss 时必须保留的数据背景。

## 真机首版的阶段验收与预算状态

采用“可扩展预算＋固定诊断＋阶段快照”，不用单一 loss 或计数自动宣告训练完成。Token 的 500/100 和真机每轮 4 局来自本轮确认；其余更新频率、batch、学习率等仍为实现候选，不是论文要求或已运行配置。

| 阶段 | 数据和训练对象 | 预算或复核单位 | 该阶段要得到什么 |
| --- | --- | --- | --- |
| token 表示 | 现有 5 条数据的缓存特征；只训 token encoder/decoder | 已确认：先预算 500 更新，每 100 诊断；有依据再延长 | 重建改善，且 decoder 确实使用 token |
| actor 模仿初始化 | 同批观测的 VLA reference；只做 actor BC | 离线分段训练；候选每 100 次 actor 更新复核，不再硬设必须完成 500 次 | 经实际部署变换后的关节动作、夹爪决策接近 reference |
| reference replay | 原 π0.5 自主执行、人工标注 | 已确认：每轮 4 局；轮末统计有效 transition、标签和轨迹，再决定是否继续采集 | 可用于 TD 的真实动作、后继状态、终止结果 |
| Q 与强 BC actor 预热 | 已标注 replay；双 Q 和 actor 联合更新 | 候选每 100 次 Q 更新复核，actor 按 2:1 更新；由指标决定是否需要更多更新或下一轮 reference | Q 开始传播结果信息，actor 未因尚不可靠的 Q 丢掉参考行为 |
| actor 执行验证 | 固定已预热 actor；建议用确定性输出 | 已确认：每轮 4 局；轮内不换权重，轮末复核 | 从“能模仿张量”进入“能闭环执行任务” |
| 在线 RLT | actor rollout → 标签 → replay → A/C 更新 | 已确认：每轮 4 局后集中更新；候选每新增 transition 目标 1–4 次 Q 更新，A:C=1:2 | 更新、保存、重载、再执行完整；效果另看任务结果 |

独立 BC 初始化是本机便于逐阶段验收的工程拆分，不是论文或 donor 强制阶段。它在已有观测上完成，不依赖新真机回合；首版按上表顺序执行，避免同时改变多个阶段。Q 预热及在线训练仍按指标复核，不设达到某个更新次数就自动接管的条件。

## 每轮交互、日志与判定

一轮的约定为：`锁定策略版本 → 逐局复位/开始/执行/标注 × 4 → 汇总 → 离线更新与诊断 → 保存 → 记录决策 → 下一轮`。Token/BC 阶段不走这个真机循环，训练进度按 optimizer 更新数记录。

每局只需在采集风格的前台终端操作，不要求每局返回聊天逐项确认；成功/失败由现场标注，助手依据完整记录复核训练，不用模型推断替代人工结果。硬件异常/中止也写日志，与正常失败分开统计；4 局记录的是尝试次数，不是 4 条成功样本。

以下为原日志规划；本版实际文件名/路径见[日志与恢复](USAGE.md#日志与恢复)。当前尚未生成正式训练记录：

| 记录 | 内容与落点 |
| --- | --- |
| 每局原始记录 | `logs/rlt/<run_id>/episodes/` 的索引/事件，关联外置数据根的图像、q/TCP、reference、实际下发动作、实际长度、末态与人工标签；每局立即完成持久化 |
| 每轮汇总 | `logs/rlt/<run_id>/rounds/`：阶段、round_id、4 局 episode_id、策略/特征/norm 身份、success/failure/aborted/pending 数、时长、有效 transition 数 |
| 训练曲线 | `metrics.jsonl`：token 正常/置零/打乱重建误差，BC 和物理关节/夹爪指标，TD error、Q/target Q、双 Q 分歧、actor 偏离，A/C 各自更新数及显存/内存 |
| 恢复进度 | `run_state.json`：当前阶段、轮次、已完成局数、标注状态、replay 游标、检查点、等待执行还是等待复核；不把待标注样本先当失败训练 |
| 决策摘要 | 正式训练开始后追加 `docs/plans/pi05-rlt/EXPERIMENT_LOG.md`：输入 checkpoint、数据范围、更新前后指标、选定输出 checkpoint、下一步及依据；原始大文件不入 Git |

轮末由助手结合曲线和现场结果选择“继续训练 / 再采同阶段 4 局 / 调整参数 / 进入下一阶段 / 结束并归档”，先写明依据和配置变化。4 局是复核节奏，不是成功率门槛、总局数上限或收敛证据；不把一轮全成功直接写成已经稳定或已经优于基线。

### Token：低重建 loss 不等于表示已经有效

固定从现有 5 条轨迹的抓取、抬起、移动、释放阶段取诊断样本，关闭随机增强，记录有效位置的重建误差。比较正确 token、置零 token、跨样本打乱 token 的结果；正确 token 应稳定更好。AR teacher forcing 会把前序真实特征送入 decoder，因此只看总重建 loss，可能掩盖 decoder 对 token 的忽略。

验收看“确实改善＋确实使用瓶颈”，再结合曲线是否趋稳；不预设跨模型通用的 MSE 数值。本轮不要求另建泛化数据集，这些检查服务于当前小数据拟合。

### Critic：有用的价值信号，而不是要求 loss 归零

TD target 会随 actor、target Q 和 replay 分布变化；在线 Q 不是一个标签固定的普通监督回归。全零奖励下，预测全零也可能让 TD loss 很低，所以单一 loss 不能证明 Q 学到了任务。

至少记录 TD error、Q/target Q 的均值和范围、双 Q 分歧，以及终止样本的预测。按真实奖励与折扣计算各条已记录轨迹的回报，检查信用是否向前传播、是否存在明显的量级错误。行为轨迹回报可做校准参考，但并不是持续变化的目标策略 Q 的精确真值。

如果数据包含自然成功与失败，可以观察是否学出差异；只有成功数据时，不把缺少失败区分能力当作数据管线故障，也不能用增加更新次数替代缺失的信息。固定终端成功 +1、失败 0 的定义下，理论折扣回报在 0–1，模型本身不保证输出有界；数值异常应追查奖励、终止和 bootstrap，而非盲目延长预热。

### Actor：先看模仿，再看行为是否改善

BC 阶段应测部署路径下的关节误差，报告 rad 单位的均值及高分位误差，并单列夹爪开闭一致性/释放时序。不能只比较 MLP 的 tanh 前输出，因为 donor 的部署动作经过 tanh；其独立 `sft_forward()` 与部署路径并不完全相同。[^7]

在线阶段分别看 BC 项、Q 项、相对 reference 的改动和实际成功/时长。actor 总 loss 可以因 Q 或权重变化变得更负；这不是独立的效果证据。适度偏离 reference 也可能正是任务改善的来源，不应把 BC loss 永远越小越好当在线目标。

`Q(actor) − Q(reference)` 可以诊断优化方向，但两者均由当前 critic 估计，不能拿这一个差值代替真实 rollout 的效果。首次目标是最小训练闭环，优于已成功 π0.5 是另一个结论。

## 每步 checkpoint

Stage2 可以按每个 learner 更新保存；按 2:1 调度，一半快照中的 actor 权重会与前一步相同，但 Q 已变化。

按 donor 实际 Linear/LayerNorm 层静态计数：actor 有效分支 731,532 参数；父类还保留 35,980 个未被固定 std 路径使用的 logstd 参数；每个 Q 为 697,345 参数。合计 2,162,202，FP32 权重约 8.25 MiB。连完整 target 模型和 Adam 两状态，张量主体约 33 MiB，不含 replay、序列化元数据与冻结 VLA/token。这是结构估算，不是实测 checkpoint 文件大小。[^7]

当前 donor 的 `runner.save_interval=1` 是每个 runner cycle 保存，而非每个梯度更新。一次 `run_training()` 最多可做 400/1,600 个更新；现有保存函数还会调用 replay checkpoint。真正的逐更新保存需要在 learner 更新循环增加明确的保存点，不能只改 YAML 的 `save_interval`。[^4][^8]

建议的保存分层：

- 每个 learner 更新：actor、双 Q/target、optimizer/scheduler、随机状态、计数、replay 游标与配置身份，构成配套恢复点；同时记录当时的指标。
- 每局数据：原始 episode 追加持久化，保存标签状态；每轮 4 局汇总后再集中更新，不在每个梯度步重复复制图像和全部 replay。
- 冻结 VLA/token：独立保存一次，以不可变路径/hash 引用；Stage1 token 大模块不采用小头的逐步全量保存策略。
- 高频快照保留最近一段，并永久保留阶段节点/选定模型；如果选择全保留，10,000 个约 33 MiB 的恢复点约为 322 GiB，需要明确磁盘预算。

先落盘不可变快照，再发布完成标记。阶段切换与 rollout 同步以完整保存的版本为单位；4 局内固定 checkpoint，下一轮才装载新版本。

## 终端交互在 RLinf 中的位置

交互主体属于 EnvWorker 管理的环境/episode wrapper；模型推理和网络优化分别保留在 rollout worker 与训练 worker。donor 已有 `KeyboardRewardDoneWrapper` 和 `KeyboardRLTPolicySwitchWrapper`：前者产生 reward/done，后者发出进入 actor 阶段的标志。这证明接入边界现成，但 donor 的 evdev 全局物理键盘监听不是本项目的前台终端交互。[^9]

| 层 | 责任 | UR5 首版接法 |
| --- | --- | --- |
| 前台终端 | 收取复位/开始/结束/结果，显示阶段与版本 | 复用 `TerminalKeyPoller` 和现有 s/f/a 结果标注习惯 |
| EnvWorker → UR5Env / HumanEpisodeWrapper | 相机/机器人、reset、动作执行、reward/terminated/truncated | 通过本机 IPC 接收带 episode_id 的终端事件 |
| RolloutWorker | 调用冻结特征模型、reference、小 actor，选择执行策略 | 接 JAX feature provider；整轮 4 局固定 reference 或 actor 版本 |
| ActorWorker / learner | 从 replay 更新双 Q 和 actor、保存、发布权重 | 4 局后集中更新；复用 RLT 损失；这里名叫 ActorWorker，但也训练 critic |

现有 `EnvWorker.env_interact_step()` 已调用环境的 `chunk_step()`，rollout worker 通过 `rlt_feature_model` 和 route 生成动作。UR5 应替换环境/episode 适配，并新增 JAX 特征桥；并不是只换一行 EnvWorker 配置就已经完成整个接入。[^10]

建议前台流程为：

`复位确认 → 开始 → 策略自主执行 → 结束/停止 → s/f/a 标注 → episode 入库 → 下一局；累计 4 局后，统一更新/复核/保存 → 下一轮`

沿用终端按键和结果日志，不复用采集时的 freedrive 控制循环；在线 rollout 不要求重新拖动示教。人工结果回填到同一局的真实终止 transition，最终观测在复位之前保存；等待人标注的时间不虚构成机器人动作或额外 RL transition。已有采集终端和结果入口可直接参考。[^11]

完整任务的 reference 预热也必须进入 replay。donor 的 `full_task` route 已将训练模式下的 reference 记录与 actor 接管解耦；legacy 真机 route 把记录绑定 actor_switch，若直接套用会丢掉预热数据。首版应保留 full-task 语义，不增加关键阶段的手动接管按键。[^5]

## Sources

[^1]: Charles Xu et al., [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/html/2604.23073v2), v2, 2026-04-30，Algorithm 1、§V、Appendix B/C。
[^2]: Yutenji-Nyamu/rlinf_fastwam，固定 `d3acd650`：[Stage1 配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml)、[AR token 模块](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/modules/rlt_token_transformer.py)。
[^3]: 同一固定版本：[Stage2 基础配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current.yaml)、[8env250 覆盖配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250.yaml)。
[^4]: 同一固定版本：[RLT learner](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)，`_actor_objective_weights`、`_rlt_updates_to_run`、`run_training`；[父类 SAC worker](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/workers/actor/fsdp_sac_policy_worker.py)，`update_one_epoch`。
[^5]: 同一固定版本：[RLT route](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/algorithms/rlt/route.py)，`FullTaskRLTRoute`、`RealworldRLTRoute`、`build_rlt_route`。
[^6]: 本地项目：[成功记录](../pi05/PHYSICAL_RESULT_20260910.md)、[方块接入规划](CUBE_PLAN.md)。本稿不重新执行机器人验证。
[^7]: 同一固定 donor：[RLTMLPPolicy](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py)、[MLPPolicy](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/mlp_policy/mlp_policy.py)、[QHead](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/modules/q_head.py)、[make_mlp](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/models/embodiment/modules/utils.py)。
[^8]: 同一固定版本：[EmbodiedRunner](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/runners/embodied_runner.py)，`run`、`_maybe_eval_and_checkpoint`、`_save_checkpoint`；父类 SAC worker 的 `save_checkpoint` 保存模型、optimizer、target、replay，RLT 子类追加阶段计数/契约。
[^9]: 同一固定版本：[reward/done wrapper](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/envs/realworld/common/wrappers/reward_done_wrapper.py)、[policy-switch wrapper](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/envs/realworld/common/wrappers/keyboard_rlt_policy_switch_wrapper.py)、[evdev listener](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/envs/realworld/common/keyboard/keyboard_listener.py)。
[^10]: 同一固定版本：[EnvWorker](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/workers/env/env_worker.py)、[RolloutWorker](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/d3acd650869d376c14c1406d90865ef90d680432/rlinf/workers/rollout/hf/huggingface_worker.py)。
[^11]: 本地项目固定审查起点 `483a612`：[前台按键](../../../src/ur5e_real/collection/terminal.py)、[采集结果标注](../../../src/ur5e_real/operator.py)。源码与论文核对日期：2026-09-10。
