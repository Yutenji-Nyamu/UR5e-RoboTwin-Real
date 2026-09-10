# π0.5 真机推理参数：来源、当前值和验证边界

后续：2026-09-10操作者确认本页K20配置的[方块真机成功](PHYSICAL_RESULT_20260910.md)；
下文09-09“未进行试跑”保留为当时开发验证边界，参数不因本次文档更新而改变。

2026-09-09。依据锁定 RoboTwin `210720340637cb4619283b295dde4cdd807c9e66`、
已完成实验 `joint5_sft_20260909_01/1000` 的 recipe/contract，以及当前本地代码。
用户确认首次真机执行 **K=20**；本轮只调整默认值、软件测试与文档，不启动模型服务或硬件。
后续同日按用户要求新增两条操作短命令，见 [USAGE](USAGE.md)；未改变下面的模型/动作参数，未进行真机试跑。

## 1. 先分清H、K和时间

- H=50：训练及模型每次预测的动作点数，保留不变。
- K=20：用户本轮选定的实际执行前缀；执行完后重新观测/推理，不继续消费旧预测的后30点。
- 10Hz：这批UR示教重采样后的时间语义；20点名义2秒，50点覆盖名义5秒。
  同步推理等待、限速重定时和调度开销会增加墙钟时间；提前释放停止时可能不足20点。
- 500Hz：UR joint servoJ 的插值/发送频率，不是模型推理频率。

原生 [deploy_policy.yml](../../../.third_party/RoboTwin/policy/pi05/deploy_policy.yml) 写 `pi0_step: 50`，
[deploy_policy.py](../../../.third_party/RoboTwin/policy/pi05/deploy_policy.py) 取 `actions[:model.pi0_step]`。
本地原K=6是适配时继承DP短chunk的工程初值，**不是π0.5训练要求或原生默认**；此前直接推荐6的依据不足，现由用户决策替换。
K小于H不需要重新训练；不能反过来认为H=50强制K=50。

同一vendor附带的 [OpenPI ALOHA真机示例](../../../.third_party/RoboTwin/policy/pi05/examples/aloha_real/main.py)
另用K=25、运行上限50Hz；这不是RoboTwin仿真部署或本机UR配置。
[RoboTwin数据转换示例](../../../.third_party/RoboTwin/policy/pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py)
标注fps=50，而本批UR导出明确为10Hz；不能在推理时换成50Hz、把已学动作时间压缩五倍。
RoboTwin仿真的 [take_action](../../../.third_party/RoboTwin/envs/_base_task.py) 还经TOPP重定时、250Hz仿真步进，
其动作点数不能直接解释为本机UR的墙钟时间。

## 2. 模型和训练契约：推理必须保持一致

| 参数 | 当前值 | 来源与理由 |
| --- | --- | --- |
| 模型/权重 | 原生π0.5，JAX；本次1000步SFT | `pi05_base`架构；checkpoint是本批5条数据训练结果，不是DP或未微调基础模型 |
| 预测窗口H | 50 | 原生 `Pi0Config.action_horizon`，本批数据及checkpoint contract也固定50 |
| 动作点间隔 | 0.1秒 / 10Hz | 本地示教重采样与训练标签；不是π0.5的通用固定频率 |
| 观测历史 | 当前1帧 | 原生π0.5接口＋本次训练，无DP的3帧历史堆叠 |
| 相机/图像 | head＋右腕RGB；224×224模型输入；缺左腕mask=false | 两实物相机来自UR硬件；resize与缺相机机制来自原生变换，训推一致 |
| 物理动作 | 6轴关节角rad＋夹爪 | 用户选择joint-space；标签是实测下一时刻q，不是TCP、速度或机械臂当时收到的控制指令 |
| 兼容维度 | 外部`[q6,0,q6,g]`14维，模型pad32 | 14维封装是本地兼容设计；32维是原生模型。执行第一组q6和第13列g，不混合重复预测 |
| 动作变换 | 内部delta相对本次q，输出恢复绝对q | 原生ALOHA delta/inverse机制被本次训练启用；执行端不得再次累计delta |
| ALOHA标定 | `adapt_to_pi=False` | UR不使用ALOHA关节符号/夹爪标定，属于本地显式覆盖；并非原生ALOHA默认值 |
| 归一化 | 本批数据的quantile统计 | 原生归一化算法；统计由本批UR数据计算并随checkpoint保存，服务校验哈希，不借用DP/ALOHA统计 |
| 提示词 | `Pick up the cube and place it back on the table.` | 本批数据固定任务文本，训练/在线使用同一prompt |

源码：[模型默认](../../../.third_party/RoboTwin/policy/pi05/src/openpi/models/pi0_config.py)、
[原生数据/模型变换](../../../.third_party/RoboTwin/policy/pi05/src/openpi/training/config.py)、
[UR训练配置](../../../src/ur5e_real/adapters/robotwin_pi05/config.py)、
[数据导出](../../../src/ur5e_real/adapters/robotwin_pi05/dataset.py)、
[契约](../../../src/ur5e_real/adapters/robotwin_pi05/contract.py)。
本地权重内 `ur5e_recipe.json` / `ur5e_contract.json` 是该次训练实参证据；不修改旧数据或checkpoint。

## 3. 部署、执行与设备参数：不冒称模型原生要求

| 参数 | 当前值 | 来源、可调性与边界 |
| --- | --- | --- |
| 执行前缀K | **20**，CLI `--action-steps` | 用户09-09选定折中；1..50可选，不是DP的6或原生部署默认50；尚未实机验证最优性 |
| flow采样步数 | 10，服务端 `--diffusion-steps` | 原生 `sample_actions(num_steps=10)` 默认，本次离线评估也用10；不是因为DP恰好也用10，不影响H/K |
| 每次推理batch | 1 | 在线一次当前观测；训练batch=8不照搬到机器人 |
| 闭环方式 | 同步推理→执行K点→再观察 | 本地执行编排；推理期间servoJ保持最后目标，未实现RTC/异步动作拼接 |
| 轨迹处理 | 逐关节线性插值，保持端点 | 本地UR实现；不是复制RoboTwin的TOPP规划或DP的TCP插值/IK |
| joint限速 | 上限0.6rad/s，CLI `--speed` | 本地软件限速；示教审计约0.52rad/s量级支撑初值，非原生模型参数/硬件认证值，也不是DP的0.40m/s |
| 重定时 | 最多2倍；到点误差0.08rad；调度落后上限0.1秒 | 本地工程阈值，超限拒绝/停止；未通过实机调优。`--speed`不是无条件播放倍率 |
| 伺服频率 | 500Hz / 0.002秒 | UR e-Series servoJ控制周期，本地复用已有控制层；不是训练采样频率 |
| servoJ lookahead/gain | 0.1秒 / 300 | 与UR官方默认及已有DP控制层值一致；保留作为控制初值，不宣称对新joint策略已调优 |
| 回原位 | 5条示教起始q均值；0.1rad/s；对齐误差0.03rad | 目标来自数据；速度/容差是本地工程值，不是DP的TCP home。限本demo包络，最长30秒，不承诺任意姿态归位 |
| 空间包络 | 示教q范围＋0.15rad；TCP XYZ范围＋0.05m | 实测范围加本地选定裕量；TCP offset按数据契约检查，不是整臂碰撞模型 |
| 跟踪/状态/命令期限 | 0.15rad / 0.5秒 / 2秒 | joint控制器工程阈值；2秒是距最近目标更新的时限，不是整个chunk时限，流式执行每2ms更新目标 |
| 相机采集 | 640×480、30fps；启动60帧预热 | 实物设备的`lab.yaml`配置，与训练10Hz及模型224px输入分别处理 |
| 相机缓存期限 | 0.2秒；首帧等待2秒 | 本地实时性检查；是交付缓存的新鲜度检查，不等于图像/q硬件同步保证 |
| 模型通信 | 手动服务默认localhost:8005；短命令自动选临时回环端口；RPC最长1.5秒；服务预热2次＋运动前live-shape预热 | 本地进程隔离/实时性选择；短命令另设600秒加载/预热期限，退出清理自有模型进程；RPC期限低于2秒命令期限，真实warm p50/p95仍待测 |
| 夹爪含义/阈值 | 0开/1关；关≥0.60、开≤0.40 | 0/1来自本地采集语义；阈值来自共享稀疏夹爪策略，不是π0.5模型硬编码或实测开口反馈 |
| 夹爪去抖 | 连续2点；命令间隔≥0.5秒；一次循环 | 本地π0.5 MVP时序覆盖；DP默认是连续3点/2秒。未从训练推导或做新实机调优 |
| 结束/上限 | close→open后hold1秒停止；最多30 chunks | 本地试跑逻辑，不是任务成功识别。K20时上限600策略点、名义60秒（通常提前结束）；保持此上限，本轮不另改运行时长 |

源码：[推理入口/夹爪接入](../../../src/ur5e_real/adapters/robotwin_pi05/infer.py)、
[原生flow采样](../../../.third_party/RoboTwin/policy/pi05/src/openpi/models/pi0.py)、
[服务](../../../src/ur5e_real/adapters/robotwin_pi05/serve.py)、
[关节插值](../../../src/ur5e_real/control/joint.py)、[servoJ控制](../../../src/ur5e_real/control/joint_servoj.py)、
[home与相机](../../../src/ur5e_real/adapters/robotwin_pi05/runtime.py)、
[RPC](../../../src/ur5e_real/adapters/robotwin_pi05/client.py)、
[夹爪状态机](../../../src/ur5e_real/control/gripper_policy.py)、[设备配置](../../../configs/lab.yaml)、
[示教审计](JOINT5_DATA_AUDIT.json)。
UR语义参见 [官方servoJ文档](https://www.universal-robots.com/manuals/EN/HTML/SW5_20/Content/prod-scriptmanual/G5/servoj_qavt0-008lookahead_time.htm)。

## 4. 不混淆历史指标、训练配置和执行结果

- 已有500/1000步评估的 `prefix_*` 固定指**前6点**，不能改标签称为前20点效果。
  `evaluate.py` / `evaluate`入口仍保留这项历史训练诊断；`action_metrics(..., first_steps=20)`可明确计算20点指标，
  但本轮未加载模型重评，前20点真实指标尚不存在。全H有效目标指标仍可独立参考。
- 本轮代码每个live/shadow chunk记录 `action_horizon` / `action_steps` / `policy_hz`；
  `executed_waypoints`才是实际完成点数，可能因close→open提前停止而小于K。
- batch8、workers2、expert-only、LR2.5e-5、warmup100、日程3000、1000步结束、每500保存，
  都是本轮训练选择，不是原生部署参数。原生示例full/batch64/20000步不代表本次实参。
  图像增强关闭、EMA关闭也是本地训练覆盖，详见 [训练记录](TRAINING_RUN_20260909.md)。
- 用户同意的“数据尾部至少保留open后1秒”是数据裁剪要求；推理端“开爪后hold1秒并停止”是另一项本地运行逻辑，
  不能把前者当作模型学出了自动成功判定。
- 已新增短命令封装，未新增完整录像/成功标注流程，未连接机械臂、相机或夹爪，未启动SFT或真实模型服务。
  可执行短命令/模块命令见 [USAGE](USAGE.md)；DP的`ur5e-infer`仍不接受π0.5目录checkpoint。

## 5. K=20变更验证（短命令封装前的快照）

- 完整无硬件回归（包含原生localhost协议的假模型往返）98项通过，Ruff及diff空白检查通过。
- 新增6项检查覆盖CLI默认20/显式覆盖、50点预测只取前20、每段日志的H/K/频率、
  假控制器20点对应1000个servo点/名义2秒，以及保留历史前6点评估语义。
- 本页及README/USAGE/DECISIONS的45个本地链接均可解析。
- 回归发现3份旧训练摘要/文档的本机绝对路径不符合已有仓库规则；仅改为仓库根相对路径或按环境名调用Python。
  两份JSON新增 `paths_relative_to=repository_root`；结构化对比确认所有历史参数/指标不变，原始日志/权重未动。
- 上述均为软件检查，不是模型实际闭环或真机验证；未改变DP、训练数据、checkpoint或其原生代码。

## 6. 同日新增操作短命令

- [操作层实现](../../../src/ur5e_real/pi05_operator.py) 自动定位仓库和checkpoint配套数据；
  init调用joint prepare，infer在隔离Python中启动模型并核对实例身份，再进入原有硬件推理循环。
- 已安装 `ur5e-pi05-infer-init` / `ur5e-pi05-infer`，与DP一样通过系统PATH直接调用。
  旧DP入口不改；模型子进程只在实际infer/shadow时启动，退出时仅清理该命令自己的进程。
- 41项针对性检查通过：短名解析/歧义拒绝、参数透传、无硬件dry-run、启动失败/中断清理、
  错误服务实例拒绝、旧DP回归与仓库规则。两个已安装入口均从项目外用真实checkpoint完成dry-run。
- 本轮没有启动真实模型或机械臂；自动模型加载/预热及真实joint执行的现场结果仍待首次使用。
