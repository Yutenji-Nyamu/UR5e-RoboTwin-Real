# RLT 实施记录

2026-09-10：按授权实现并做离线/假环境检查。**软件链已具备，未启动正式 token/BC/A/C 训练或 RLT 真机回合。**
冻结基线 `joint5_sft_20260909_01:1000`；旧 TCP DP 和成功的 π0.5 入口不变。
操作见 [USAGE](USAGE.md)，正式进度追加 [EXPERIMENT_LOG](EXPERIMENT_LOG.md)。

## 实现分层

| 层 | 文件与职责 | 来源/变化 |
| --- | --- | --- |
| 原生 VLA | `robotwin_pi05/rlt_features.py` | 同一次prefill返回reference及最后一层image prefix/mask；保持JAX采样，不转换权重 |
| Token | `rlinf_rlt/vendor/`、`cache.py`、`train.py` | donor AR原文件；缓存后卸载VLA，只训练token |
| A/C | `models.py / learner.py` | donor三层tanh actor、LayerNorm双Q、BC−Q、Adam、2:1与EMA |
| 真机规范动作 | `actions.py / serve.py` | q6+gripper、可逆canonical域、一次反变换；独立PyTorch服务 |
| Env边界 | `environment.py / rounds.py / replay.py` | 原joint执行器、终端复位/结果、实际长度、复位前末态 |
| 协调/恢复 | `run.py / checkpoint.py / rlt_operator.py` | 4局固定版本、轮末更新/复核、原子保存、资源/命令/决策日志 |

除另列者，模块位于 `src/ur5e_real/adapters/rlinf_rlt/`。
本版**对齐RLinf算法，不运行完整Ray/FSDP/多仿真环境runner**：本地单设备coordinator替换调度循环，
保留Env / rollout / learner的责任边界；底层设备不导入模型框架。不是“只换EnvWorker就完成全部接入”。

## 参数来源

donor锁为 `d3acd650869d376c14c1406d90865ef90d680432`，源文件/配方/许可证见
[donor.lock.json](../../../integrations/rlt/donor.lock.json)。

| 项目 | 本版值 | 对齐与真机变化 |
| --- | --- | --- |
| Token | input/z2048、768位置、各2层、8头、MLP ratio4、AR teacher forcing、有效mask MSE | donor原文件，SHA256测试锁定；本机实测512有效位置，空左腕256位置无效 |
| Token optimizer | AdamW lr2.5e-5、β=.9/.95、eps1e-8、wd1e-10、clip1 | donor token-only配方 |
| Token schedule | warmup100、cosine到10%；首段500、每100诊断/保存 | 曲线形状对齐donor；500与周期诊断由用户确认，代替donor2000步；batch2为本机候选 |
| A/C | 各3×256；Adam lr1e-4、β=.9/.999、eps1e-8、clip10；τ=.005 | donor源码/Stage2配方，不引入entropy/alpha优化 |
| TD/actor | 当前actor产生next action，target双Q取min；actor用Q1；reference dropout .5 | donor；Q:A=2:1，第1、3、5…次Q更新同时更新A；每次Q更新后EMA |
| BC/Q权重 | warmup7/.05；online逐渐到2.5/.45，ramp10000 | donor数值；online起点改为明确进入该阶段的实际更新，不照搬仿真固定warmup计数门 |
| Offline BC | 确定性actor拟合reference | 本机独立初始化；后续联合预热仍用stochastic BC−Q |
| 宏动作 | H50/K20、目标10Hz、10 flow steps | 成功基线/用户K20决定，不照抄donor K10/4 flow steps |
| 执行 | servoJ500Hz、限速.6rad/s、一次close→open、释放后hold至少1秒 | 沿用真机joint执行器及现有关节/TCP边界 |
| 探索 | reference不加小头噪声；actor_probe确定性；online tanh前std=.002 | donor起点，**不是.002rad**；诊断按各关节与夹爪尺度换算，首次online前复核 |
| A/C预算 | batch32，每次显式 `--steps N` | 本机候选/工作预算；日志报告更新数与数据复用，建议1–4更新/新transition不是通过线 |
| 回合组织 | 4局串行、同版本/配置；轮末训练及明确复核 | 用户决定，不自动按仿真500transition/5000update接管 |

### 动作与终止

从原生32D取 `[0:6,13]`，各轴除以离线reference最大绝对值的1.25倍、缩放下限1.25。
得到固定7D canonical域；越域reference要求复核，不悄悄clip训练目标。
逆过程乘回scale、使用相同quantile inverse、仅对q6加一次chunk起始q，夹爪保持 `[0,1]`。
输出回填14D兼容协议，仅K点进入joint执行器。

Q输入完整请求K×7，实际 `k≤K` 用于 `Σ γ^i r_i + γ^k Q(next)`（γ=.99/策略动作点）及BC前缀mask。
**不把事后执行长度k输入Q**；不会把终止结果泄漏为动作特征。
500Hz插值、推理等待和固定释放hold不是新增策略动作点，墙钟与轨迹另记。
这是策略点折扣，不是墙钟折扣；若改用后者需另建实验版本。

proprio13为原生归一化q6+gripper，再加执行器phase one-hot、两个阈值计数与冷却余量。
SFT缺少执行器历史：缓存标明unobserved，离线BC用辅助值扰动教actor忽略无关历史，并另报3种辅助输入诊断；
这些合成值**不做Q replay**。真机Q使用实际观察到的执行器状态。

### 交互、记录与身份

- reset/start/result位于Env边界；learner不连接机器人，模型服务不读键盘，不使用donor全局evdev监听。
- 双图、状态/时间、reference、规范请求、500Hz producer命令、逐waypoint实测q/TCP与夹爪事件按chunk落盘。
  producer命令不冒充逐包硬件执行确认；动作异常留存部分trace。
- 末态在复位前捕获；停止控制后收s/f/t/a。成功/失败terminated；纯预算耗尽可truncated；aborted/pending不入TD。
  人工标注等待不计入policy完成时间。恢复只补剩余局，补标不重执行，不为凑成功数加局。
- 每更新保存A/C、target、两optimizer、RNG、计数/replay来源；保留最近20个和全部被round/decision引用的小头。
  Token大恢复点每100保存；原子发布、weights_only重载、身份校验；回退同时选择配套learner状态。
- 当前大文件在Git忽略的 `logs/rlt/<RUN>/`，不自动移到4TB盘。资源2秒采样、30秒摘要，额外记录每更新Torch显存/内存。
- init保存基线params逐文件SHA256、尺寸/mtime及norm/recipe身份；缓存/执行前检查清单、尺寸/mtime与norm，
  不每次重新读取5.8GB权重计算全量哈希。命令日志记录git commit和实际RLT源码摘要。

## 检查证据与当前边界

- RLT专测21项通过：完整假环境阶段流、实际k/折扣/mask、终止/截断、失败/中止/待标注、暂停续局、补标不重执行、
  固定版本、checkpoint/RNG/optimizer恢复、source SHA、独占lease，以及复用执行器的释放hold/轨迹回调。
- 原生dummy固定噪声新旧动作差 **0.0**，prefix768×64；成功SFT固定噪声差 **0.0**，prefix768×2048、512有效位置、raw action50×32。
  后者单次热特征前向约 **0.164秒**，仅一次离线采样，不能当跨进程p95或真机时延。
- `.venv/rlt`按29个锁定依赖安装、pip check通过；A6000/Torch2.4.1+cu121通过非零梯度小模型检查。
  服务导入不加载JAX；硬件命令导入不加载JAX/Torch/设备驱动。全仓库最终回归数见[实验日志](EXPERIMENT_LOG.md)。
- `ur5e-rlt`短命令已安装；初始化 `cube_rlt_20260910_01` 后，status/dry-run通过，仍在cache阶段。

下一步才是缓存、全尺寸token几步显存检查与500更新/100诊断。
完整新JAX→PyTorch→硬件跨进程延迟、真实4局交互、正式token/A/C拟合与RLT效果尚待后续验收；
不能把假环境通过写成机器人已跑通。首次reference前明确成功区域，首次online前复核各轴探索幅度。
本版不含跨run恢复点迁移或异步更新；变更不可变配置需另建实验。
