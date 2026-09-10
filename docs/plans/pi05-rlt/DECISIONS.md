# 决策、开放问题与续接记录

2026-09-10续接改为[方块RLT分阶段规划](CUBE_PLAN.md)：π0.5 joint真机已经成功，
当前只重新规划cube RLT；旧G01/G02 charger问题不再要求决策。

更新：2026-09-07。与 [主规划](README.md) 和 [证据](RESEARCH.md) 一起使用。

**本文件为第一轮历史记录，不再是当前继续入口。** 最新用户决策：只做 RoboTwin 原生 π0.5 方块最小 demo；**首版 joint-space，接受先补采 q+TCP**。当前请读 [独立决策记录](../pi05/DECISIONS.md)。旧 D02 后端建议撤回；D03 首版 TCP 被覆盖；charger/RLT 与泛化要求暂缓，不再询问旧 G01/G02。

## 1. 用户已经确认

| ID | 决策 | 依据/边界 |
| --- | --- | --- |
| U01 | 先 `pick_place_cube` 跑通 π0.5，再 `charger_plug` RLT | 用户选择；不再重复询问 |
| U02 | 终端人工复位、开始、成功/失败、停止；首版不做连续遥操作接管 | 用户选择；不等于已经确定每个按键或成功标准 |
| U03 | 本轮广泛调研、规划、讨论并形成项目文档 | 不是本轮启动训练、运行机器人或替换环境的授权 |
| U04 | 多建立持久文档，记录信息与设计决策，抵抗上下文压缩 | 文档放本项目；继续工作先读本文件 |
| U05 | 尽量自动完成可执行的调研/文档工作，交流简洁精准 | 不反复询问已确认事项；真实未知项明确标注 |

## 2. 工程建议：尚非用户逐项批准的最终设计

| ID | 当前建议 | 原因 | 何时重评 |
| --- | --- | --- | --- |
| D01 | 干净 π0.5 base，先训练 action expert/相关投影 | 减少 ALOHA 双臂仿真动作语义遗留 | 任务基线弱时对比仿真初始化 |
| D02 | 优先测试 AR RLT 分支 legacy openpi 单一后端 | 减少 SFT→RLT 权重转换；保留已有动作/replay 修复 | 一批训练、严格加载/冻结检查失败时转 LeRobot 备选 |
| D03 | 本机 7D 绝对 TCP+夹爪契约，模型内部 padding | 与原始数据一致，不污染硬件层 | 需要更强位置泛化/大旋转时另建 delta/SO(3) 版本 |
| D04 | head/wrist 两有效 RGB 视角，第三无效视角 mask | 实际只有两相机，不把复制视角当真实独立输入 | 新增物理相机时 |
| D05 | H=50 与实际执行 K 分离 | 本机 10 Hz，长盲执行影响反馈 | 测出 warm p95 延迟后定 K/H |
| D06 | charger 首版近插口、已预抓取；末端插入先闭环 | 缩短人工回合和稀疏 reward 路径 | 用户确认初态/夹具后 |
| D07 | RLT token-only Stage 1，然后冻结 VLA/token 训练小头 | 控制显存与表示漂移；兼容已有分支路径 | 基线或 token 质量不足时考虑联合训练 |
| D08 | 单环境/单设备所有者，终端前台 IPC，回合间学习 | 不让 Ray stdin/多 worker 控制权成为障碍 | 吞吐量确有需要再做异步 |
| D09 | 首版纯 RLT，不加 DVAC/GRPO/PPO/连续人类接管 | 每个阶段的错误与收益可定位 | 工程闭环和 baseline 评估完成后 |
| D10 | 参考 warmup 也写有效 replay，actor 接管与记录解耦 | 旧 RealworldRLTRoute 默认丢弃 reference transition | 首版 fake-env 路由测试时定案 |
| D11 | RLT canonical 空间有明确可逆边界映射 | tanh 的 [-1,1] 不是 TCP 物理单位；quantile 也可能越界 | 任务边界/统计确定后锁版本 |

## 3. 待讨论/核验的 gap

| ID | 问题 | 建议默认 | 影响/最晚决定点 |
| --- | --- | --- | --- |
| G01 | charger 怎样算成功，怎样算失败/超时？ | 人工二值确认，并写出可重复的物理标准 | reward；P3 前 |
| G02 | charger 从预抓取近插口开始，还是完整拿取插入？ | 先预抓取近插口 | 数据、回合长度、夹爪；P3 前 |
| G03 | 插口固定、插入容差、接触限制与反馈能力？ | 固定夹具/低速小范围；先核验能力 | 执行策略；P3 前，不用算法掩盖机械问题 |
| G04 | 现有 charger 3-success 视频质量，是否需要二次 close？ | 逐条复核并补足合格数据 | 数据集/夹爪编码；P3 前 |
| G05 | 5 条 cube 仅最小演示，还是正式效果也限定 5 条？ | 先固定 5 条训练，其余按 episode 留出 | 实验口径；P0 可按此建议准备 |
| G06 | 基础权重/旧环境是否可复用？ | 当前可读资产优先；无法访问的旧 home 不作前提 | 下载与存储；环境准备前 |
| G07 | 单后端与 LeRobot 备选哪条实际可运行？ | 一批训练+严格加载验收选择 | 工程验证，不需要用户凭空选框架 |
| G08 | terminal-only 是否还需人工切换“参考前段→actor 插入段”？ | 近插入局部 episode 可先不用阶段切换 | route/BC/replay；P5 前 |
| G09 | warmup、探索幅度、正式比较回合预算？ | 从保守小规模集成开始，按实测 baseline 调整 | 不照抄仿真 8-env 配置；P6 前 |

## 4. 已发现的关键坑（继续时不要重新踩）

- `joint_action`/`qpos` 的名称不能证明本机是关节空间；本地 HDF5 填的是 TCP pose。
- 本机 RealSense 原始输出是 BGR；π0.5 离线/live 必须一致转 RGB。
- `clean_50` = 单任务 50 episodes；Sidney π0.5 = 50-task 仿真训练，是两件事。
- 旧 RoboTwin ALOHA 输入适配会改关节符号与夹爪标定，不可套到 TCP。
- 你的 Sidney 转换器文件名带 `openpi_rlinf`，实际目标构造的是 legacy `OpenPi0ForRLActionPrediction`；不能按文件名认定与新 vendored 后端兼容。
- legacy 加载器多处 `strict=False`；新任务必须检查 missing/unexpected keys，只放行有清单的新 RLT 参数。
- 新 RLinf SFT builder 已读路径没有看到 `train_expert_only` 的冻结动作；不能把 YAML 设置当冻结证据。最终看 requires_grad、optimizer 和一步权重 diff。
- Stage 1 token encoder/decoder 不是 Stage 2 的小 actor/critic；显存不可混算。
- AR causal 重建仍需 token ablation；teacher forcing loss 降低不自动证明瓶颈有用。
- 旧 keyboard wrapper 使用 evdev，不等于支持 SSH/终端；Ray worker 通常没有 TTY。
- 旧 RealworldRLTRoute 将 reference 样本排除；有 warmup 计数不等于有 warmup replay。
- 既有 chunk runner 没有中途取消契约；不能在上层标 done 后仍执行剩余目标。
- `close→close` 是事件，不是两个不同二值状态；手动重播保真修复不等于策略学会二次收紧。
- 历史仿真成功率和旧服务器显存日志不是本机真机验证结果。

## 5. 执行账本

### 2026-09-07：完成的调研

- 核验当前仓库、RoboTwin 锁版本、DP 数据/图像/动作/执行与夹爪路径。
- 核验 GPU/内存/磁盘和原始 session manifest 计数；没有查询机器人状态或发送动作。
- 获取用户远程仓库 37 个分支 tip；重点阅读 RLT、AR、checkpoint diagnosis、π0.5/Sidney 接入和相关变体差异。
- 对照 OpenPI、RoboTwin、RLinf、LeRobot 官方源码/文档和 RLT 原始论文；把社区实现仅作补充参考。
- 用户两项任务选择已记录 U01/U02；本轮工作产物为项目 Markdown 规划，不是代码实现。
- 已写入四份规划文档并从中英文索引链接；本地文档链接检查覆盖 64 个目标，均存在。文档空白差异已检查；未执行训练/硬件测试。

### 证据边界

- 远程阅读是 shallow branch-tip/tree/source 审查，未覆盖每个分支全部历史、日志或运行结果。
- 尚未执行 π0.5/RLT 一批训练，未测本机训练显存；没有完成 checkpoint 转换或新策略真机测试。
- `/home/ur5` 当前不可读，因此旧模型/cache 是否可直接复用仍未知。不能将权限受限误报为资产不存在。

## 6. 上下文压缩后的继续入口

1. 先读本文件、[主规划](README.md) 和 [RESEARCH.md](RESEARCH.md) 的关键结论；需要精确出处再读分支表/源码。
2. 先检查 `git status --short`，尊重其他未提交修改。基线为 `aeadb3e`；本轮只新增规划及文档索引。
3. 不再询问 cube→charger 顺序和 terminal-only 选择；尚需讨论的是 G01/G02 等真实任务定义。
4. 若用户仍在讨论：只更新对应决策/方案并说明变化。若用户要求实施：先 P0，且每个新验证结果追加日期/配置/证据路径。
5. 每阶段把“已读源码推断”“离线验证通过”“真机验证通过”分开记录；保留失败尝试和回滚路径。

研究用只读副本临时路径：`/tmp/ur5e-pi05-research-4FsuAb/rlinf_fastwam`。它没有工作树，按 `git show origin/<branch>:<path>` 阅读；临时目录可能被清理，永久依据是 [分支快照](BRANCHES.md) 中 SHA 与远程链接。不要把研究 clone 直接当已安装训练环境。
