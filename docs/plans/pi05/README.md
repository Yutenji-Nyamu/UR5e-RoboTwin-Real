# RoboTwin 原生 π0.5 → UR5e：joint-space 最小真机 demo

更新：2026-09-07，第二轮讨论后。状态：设计与数据审计完成，待下一轮实施；未启动训练、采集或机器人动作。

本目录是当前 π0.5 的独立上下文，优先于旧 [π0.5/RLT 联合规划](../pi05-rlt/README.md)。继续工作先读 [决策与实施顺序](DECISIONS.md)，原生源码逐项对照见 [适配审查](ADAPTATION.md)，旧方块数据见 [数据审计](DATA_AUDIT.md)。

## 1. 当前用户决策

- 只先做 `pick_place_cube` 的 π0.5 真机最小跑通；charger、RLT、泛化评估均不进入当前验收。
- 明确选择 **RoboTwin 原生 `policy/pi05`**，参考已经工作的 DP“原生模型 + 本地适配 + 真机执行器”结构。
- 用户已确认：**首版采用关节动作，接受先补采数据**。新采集同时记录实际关节角与 TCP，保留以后选择表示的能力。
- 本轮只做深入审查、讨论、文档，并提交/推送 Git；下一轮再实施。

上一轮把 ALOHA 接口需要适配作为优先转向 RLinf/LeRobot 后端的理由，不符合当前最小目标。接口适配和训练配置调整正是本仓库 DP 已采用的方式；本轮已撤回该后端建议。

## 2. 现有 DP 到底学的是什么

```text
现有 DP
实际 TCP + 夹爪 → 14D 兼容数据 → 预测下一段绝对 TCP
                                      ↓
                  TCP 插值/限速 → UR 逆运动学 → joint servoJ

首版 π0.5（本轮确认）
实际 joint + TCP + 夹爪 → 选择 joint 数据 → 预测下一段绝对 joint
                                                   ↓
                                  joint 插值/限速 → joint servoJ
```

现有 DP 的采集、训练标签和模型输出都是 EEF/TCP，不是 joint；机器人最终用关节伺服，不意味着模型也输出关节。它不是错误链路。Diffusion Policy 原论文的 UR5 实验也使用末端位置命令，EEF 与 joint 都是正常的策略表示。[DP 原论文附录 D](https://arxiv.org/html/2303.04137v5)

本机旧 `joint_action/vector` 实际为 `[tcp6,0,tcp6,g]`；名字来自 RoboTwin 兼容格式。新 joint 数据必须明确区分为 `[q6,0,q6,g]`，不能仅凭 shape=14 判断两者相同。

| 维度 | TCP/EEF | Joint |
| --- | --- | --- |
| 模型目标 | 末端位置/姿态 + 夹爪 | 六个关节位置 + 夹爪 |
| 控制衔接 | 需要 IK；末端路径较直接 | 不需要输出侧 IK；直接给 servoJ |
| 关键工程点 | 旋转表示、IK 分支、奇异点、TCP 标定 | 关节顺序/单位、角度分支、初始姿态、各关节速度与整臂路径 |
| 当前旧数据 | 已有 | 原始文件缺失，无法当作实测 joint 使用 |
| 本轮选择 | 保留 DP/旧数据，不删 | 新 π0.5 主线，先双记录补采 |

不存在“joint 普遍正确而 EEF 普遍错误”的判断。本次用 joint 是用户选择，也让策略输出与 UR 底层位置伺服直接对应。

## 3. 采集兼收，训练/推理可选：怎样定义才不混淆

### 3.1 采集端不必二选一

新版本每条 robot sample 从**同一个 RTDE 数据包**记录：

- `controller_time_s`、`actual_q[6]`（弧度）、`actual_TCP_pose[6]`（米/旋转向量弧度）；
- 夹爪命令状态、事件计数及独立 open/close 事件流；两相机文件名及时间关联；
- 建议一并保存 `actual_qd`；保存本机接收时间用于排查延迟。相机已有时间戳能力应尽量落盘，但不伪称 RTDE 和两相机硬件严格同步；
- episode manifest 写 schema、字段/单位、关节顺序、工具/TCP 配置、起始 q/TCP、采集频率、结果与代码版本。

当前 collector 只订阅 pose/time；另一方面，执行器的 XML 已订阅 `actual_q` 等字段，只是 Python 控制类没有提供/保存完整状态。因此双记录是明确的字段与接口改造，不是机器人不支持。[UR 官方 RTDE 字段](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html)

### 3.2 模型表示选择发生在数据导出和训练配置

建议显式 `representation=joint` 或 `tcp`，把表示写入 dataset 和 checkpoint。首版只验收 joint 主线，兼收 TCP 是为后续使用已有样本，不要求同时实现第二套策略训练。

同一份双记录可分别导出两套数据、统计和 checkpoint；**不是同一 checkpoint 在推理时随意把 joint 切成 TCP**。推理端校验 checkpoint 的动作空间与 executor 一致，不一致应在连接控制前失败。不要从 7/14/32 的维度或文件名推断空间。

旧 TCP 轨迹经 IK 可以生成某个估计解，但没有当时的关节分支/实测误差；不冒充原始 joint 示教。本次按用户选择补采。

## 4. 首版建议契约

| 项目 | 建议方案 | 原因 |
| --- | --- | --- |
| 原生版本 | 当前 `robotwin.lock` 的 `210720340637cb4619283b295dde4cdd807c9e66` | 不顺带升级/破坏 DP |
| 基础模型 | 原生 `pi05_base/params`，JAX/Flax 训练与推理 | 沿用原生 trainer、flow matching、权重格式 |
| 可训练参数 | action expert + action/time 投影；冻结 vision/language backbone | 原生有 freeze_filter，无需更换框架 |
| 物理状态/动作 | q6 + commanded gripper1 | 当前夹爪没有实测开口宽度 |
| 首版兼容封装 | `[q6,0,q6,g]` 共 14D，模型内部仍 pad 32D | 沿用 DP 与原生 ALOHA 的结构；不以首轮 7D 清理扩大改动 |
| 执行输出 | 固定取 `actions[:K,:6]` 为 q；第 13 列为夹爪 | 不混用重复两组预测，不拿 14D 直接发机器人 |
| ALOHA 标定 | `adapt_to_pi=False` | 即使都是关节角，UR5 的轴/夹爪也不是 ALOHA 标定 |
| 动作变换 | 保留原生 joint delta：相对**本次观测 q**；输出恢复绝对 q | 数据文件仍存下一步绝对 q；禁止二次累加/逆变换 |
| 图像 | head + physical wrist，显式 BGR→RGB；缺失 wrist 置零并 mask | 用原生 AlohaInputs 缺相机机制；不把复制相机当真实新视角 |
| 语言 | 固定 `Pick up the cube and place it back on the table.` | 描述当前示教，不引入随机指令/新任务 |
| 时序 | 示教/数据 10 Hz；H=50，实际执行 K=6；servoJ 500 Hz | 继承 DP 的短 chunk；不是一口气执行 50 步 |
| 初始条件 | 补采时固定起始 q；执行前对齐同一关节姿态 | 相同 TCP 不保证相同整臂姿态 |

14D 是首版工程兼容建议，不是 π0.5 必须双臂，也不是模型 action_dim 改成 14；内部 32D 与 base 权重投影保持一致。7D 简化可以以后单独改 adapter，不影响这次 joint 决策。默认保留原生 padding/loss 行为，不为第一轮另改模型损失。

## 5. 接真机需要哪些代码

| 部分 | 可复用 | 下一轮新增/适配 |
| --- | --- | --- |
| 采集与 review | freedrive、相机、夹爪事件、manifest、人工 success/abort | 同包 q+TCP 状态；schema v3；保持旧 TCP reader 行为 |
| 数据处理 | session 对齐、源 ID、不可变导出、首尾裁剪思路 | 指定 run IDs；joint 标签；10 Hz LeRobot；RGB；事件保护尾段 |
| 原生 π0.5 | Pi0Config(pi05=True)、train.py、Policy、normalizer、checkpoint | UR5 config、expert freeze、归一化辅助脚本兼容修正 |
| 推理流程 | offline→shadow→execute、相机/状态获取、日志、串口状态机 | 原生 policy server/client；动作空间元数据检查 |
| 500 Hz 控制 | RTDE 收发线程、时钟、watchdog、prime/activate、runtime 检查 | `set_target_joints`、joint 限速/插值、直接 servoJ 脚本、当前 q 初始化 |
| 准备姿态 | 当前 prepare/home 的流程与就绪检查 | `home_joint_positions` / joint home 对齐，不只比较 TCP |

单独模型进程使用 Python 3.11+/原生依赖，硬件侧保留现有 Python 3.10 环境。借用原生 WebSocket policy 服务做同机 IPC，不要求安装/运行 RoboTwin 仿真来执行真机。

建议新增 `adapters/robotwin_pi05/{config,process_data,train,serve,infer}.py`、joint 执行适配与相应测试；路径是设计，尚未实现。注册配置和入口放本仓库，优先调用原生函数；确需修 vendor 时将最小补丁托管并纳入 bootstrap，不只手改 ignored 的 `.third_party`。

## 6. 参数初值：不需要再逐个问用户

| 参数 | 首次建议 | 调整依据 |
| --- | --- | --- |
| 成功示教数 | 先补采 5 条连贯、慢速、同摆场的 joint+TCP 成功记录 | 小样本跑通；不强制划分泛化验证集 |
| 训练 batch | 2，必要时降 1；单 GPU / `fsdp_devices=1` | 一次真实 backward + optimizer step 的显存峰值 |
| EMA | `None` | 避免首版为完整模型额外持有 EMA 副本 |
| 训练预算 | 先 10-step plumbing；再 1000 steps，按拟合结果决定续到 2000 | 这是试跑预算，不是保证在指定步数成功 |
| 学习率 | peak `2.5e-5`，warmup 100；原生 AdamW | 首版无需算法调参搜索；decay horizon 与实际训练预算匹配 |
| 保存 | 每 250 steps + final，保留独立版本/配置/数据清单 | 能重载、可比较；不开覆盖旧结果 |
| worker/log | data workers 0 起步，W&B disabled/offline | 减少多进程/GPU初始化和外部服务依赖 |
| 推理 | H=50、K=6、denoise steps=10；先热身再计端到端 p50/p95 | 需要缩短盲区时才调 K/steps；先不做 RTC |
| 关节软速度 | 每关节 0.5 rad/s **候选初值**；最大步进 0.05 rad/0.1s | 不是机器人硬件上限；以慢速新示教和受控执行核定 |
| servoJ | 500 Hz，lookahead 0.1，gain 300 | 沿用已用参数，不同时改变控制调优 |
| 首尾 | 起始运动前保留 3 帧；末尾至少覆盖最后 open 后 1 秒 | 旧图像显示 pose 静止不代表夹爪已释放 |

joint 模式不能复制 TCP 的 m/s 限速数值作为 rad/s，也不能只限制关节速度就认为整臂路径有保障：限定该 demo 的关节范围/工作区域、检查跟踪偏差与命令连续性；出现严重限幅时放慢示教或明确重定时，不让夹爪按原索引提前动作。

上述是工程默认，不扩展为新的用户选择题。本机 48 GB 首先验证 expert-only 的单步资源；不运行原生 full/batch64 默认，也不在当前阶段追加泛化/多任务目标。

## 7. 下一轮实施顺序与最小验收

1. **双记录 + joint 执行骨架**：旧 TCP 路径回归通过；假状态/假控制器测试单位、维度、prime、stop 和 q/TCP 混用拒绝。
2. **真实只读字段与起始姿态检查 → 补采**：确认每包 q/TCP/时间齐全，固定 home q，采 5 条成功；原始文件不裁剪。
3. **数据导出与训练接入**：保留动作中间停顿、close/open 和完成段；norm/inverse round-trip；原生 loader 一批；打印冻结/可训练参数，保存重载。
4. **小样本 SFT**：先 10 steps 测通，再短预算拟合；只要求当前 demo 需要的表现。
5. **offline→shadow→受控 joint 执行**：先用记录关节轨迹确认执行路径，再接模型；按所采摆场完成抓取、抬起、放下并释放，保留本次日志。

测试要区分“代码测试通过”“读到了真实 q”“记录动作执行通过”“模型真机完成任务”。目前均未新增执行结果。不要用模型能输出数组、loss 下降或 servoJ 打印日志替代最终 demo。

当前没有必须再决定的模型后端或动作空间问题；剩下主要是实现时核验 joint 起始姿态/约束、字段可用性、数据质量与运行资源。
