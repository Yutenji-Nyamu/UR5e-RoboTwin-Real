# π0.5 joint 真机接入：开发交付

后续：2026-09-09 [训练工具与数步batch测试](TRAINING_PREPARATION.md)；随后已启动 [1000步正式SFT与监控](TRAINING_RUN_20260909.md)。本页以下为9月8日交付快照。

2026-09-08；实现起点 `a3e5427`。用户已补采5条，授权实现、检查并推送。
本轮交付数据/模型/执行适配代码，**不等于完整基础权重SFT或真机demo完成**。
命令见 [操作说明](USAGE.md)，旧采集开发记录见 [采集交付](COLLECTION_IMPLEMENTATION.md)。

## 已实现

- 独立 joint 控制器、RTDE recipe、直接 servoJ 脚本；500Hz关节插值、默认0.6rad/s、
  最大2倍保端点重定时；跨chunk从上个命令接续，不向滞后实测q跳变，不追赶式突发发点。
- prime前确认停止，寄存器初始化actual_q/mode0；新脚本需本次nonce、program ID、runtime2回执；
  跟踪误差、遥测/命令失联、demo关节/TCP范围、TCP offset检查，异常或Ctrl-C停止。
- 独立慢速joint home（0.1rad/s），默认dry-run；推理只检查home、不暗中自动移动归位。
- v3实测joint数据审计、释放保护裁剪、10Hz LeRobot不可变导出与原生统计。
  保留旧TCP数据/DP/手工回放链，双记录不表示同一checkpoint可随意切动作空间。
- 原生 `Pi0Config(pi05=True)`、flow loss、init/train step、checkpoint和Policy；
  本地负责编排、动作契约、冻结/重载核验。训练/推理均采用joint表示，外部标签绝对q[t+1]，
  原生内部delta相对本次q[t]；inverse仅一次。兼容14D `[q6,0,q6,g]`，模型pad32，
  物理执行第一组q6和第13列夹爪，缺失左腕图像mask为false。
- 独立Python3.11/JAX环境与版本锁；本机WebSocket服务、offline/shadow/execute客户端。
  先warmup；完整动作/数据契约检查；超时断开、不使用迟到结果。开发smoke checkpoint不允许execute。
  一次close→open后不再执行剩余策略waypoint，稳定hold1秒后停止。

## 这5条数据

全部 `pick_place_cube / completed / success / raw schema v3`；
q/TCP/qd同包字段齐全。关键图像已查看抓起、抬起、放下和释放，作为首版固定训练候选集。

| run ID | 原始图像对 | 导出transitions |
| --- | ---: | ---: |
| 20260908_153254 | 134 | 102 |
| 20260908_153339 | 114 | 81 |
| 20260908_153413 | 103 | 79 |
| 20260908_153445 | 104 | 78 |
| 20260908_153525 | 102 | 74 |
| 合计 | 557 | 414 |

- 初始q跨session各轴差异小于0.00005rad；TCP offset均为0；实测峰值qd约0.526rad/s。
- 每条原始尾部在open后仍有1.9～2.5秒图像。导出最后**训练观测**至少保留open后1秒，
  再多0.1秒作下一状态标签，不修改原始文件。
- 全部起始有0.2秒image gap；153445中间另有一个0.2秒gap。明确重采样；
  153445复用1帧图像（0.1秒偏移），未把source当作完美均匀采样。
- 本地导出：`/data/robotics/ur5e-real/pi05/lerobot/ur5e/pick_place_cube_joint_5_v20260908`。
  数据集内保存来源/契约/统计和parquet SHA，项目记录 [机器审计](JOINT5_DATA_AUDIT.json)；
  数据盘DATA_LOG已补记首批真实v3与旧TCP分界。不再要求用户重复补采这5条。

## 实际验证结果

| 检查 | 结果与边界 |
| --- | --- |
| 核心回归＋原生loopback协议 | 84项通过；包括旧DP、关节假控制器、跨chunk连续、提前结束、过期prime/超时/空间拒绝 |
| 真实5条导出 | 414 transitions；不可变版本、SHA与原生统计就绪 |
| 原生数据检查 | LeRobot loader、15个跨episode取样的训练/RPC RGB与state/action一致、quantile/delta/pad/inverse往返、224px输入通过 |
| 隔离环境 | Python3.11.13；JAX0.5.0识别CudaDevice(0)；188包pip check通过；RTX A6000 48GB、driver580.95.05 |
| 原生dummy GPU单步 | loss/gradient有限、expert参数改变、frozen参数哈希不变，通过 |
| 原生dummy保存/重载/推理 | 参数＋step＋optimizer哈希重载一致，原生Policy输出finite(50,14)，通过 |
| full π0.5基础权重训练 | **未验证**；未下载完整基础参数，未正式SFT |
| 新joint执行器/π0.5真机 | **未运行**；只运行prepare dry-run，未连接机械臂/相机/夹爪 |

dummy结果见 [机器记录](MODEL_PLUMBING_CHECK.json)：使用原生dummy Gemma随机参数和原生vision trunk，
是编排测试，不能作为完整base训练、显存峰值或推理时延证据。
全尺寸原生架构只做了抽象参数计数：trainable430,098,464 / frozen2,923,335,408。
完整官方base参数约12.5GB；本次1MiB范围请求测速约0.38MB/s，未启动该大下载。

## 联调中修正的问题

- 原生config的3处 `local_files_only` 参数不存在，导致整个模块导入失败：托管最小删除补丁。
- 原生checkpoint callback使用了锁定Orbax0.11.1没有的Future类：改为await完成小型asset写入，
  再返回空future列表；模型参数/optimizer格式不变，已实际保存重载。
- 原生下载器未取线程future结果，可能吞掉传输失败：增加异常传播，不把失败下载发布为完整cache。
- native norm辅助脚本引用不存在的loader/错误batch接口，本地使用原生RunningStats和真实下一状态序列；
  loader的workers通过TrainConfig传入，不照抄native main中不匹配的调用参数。
- Pydantic2.10.4与numpydantic1.6.6按原生版本配对；隔离env用精确sync清掉旧解析遗留冲突。
  硬件env只新加websockets15.0.1/msgpack1.1.1，原DP/PyTorch依赖不变。

三个vendor补丁均入Git并纳入bootstrap；运行时仅接受锁定commit及这些精确补丁。
不修改native模型/损失；不把ignored vendor手改作为不可复现的交付。
失败的dummy开发产物保留在ignored环境目录，最终通过实验为 `dummy_plumbing_04`。

## 下一步（不再有后端/动作空间决策阻塞）

1. 下载并校验完整 `pi05_base/params`，做batch1真实基础权重单步；核对实际显存与加载兼容。
2. 用本次5条做最小SFT（起始1000steps/batch2，必要时batch1），再离线检查动作与warm时延。
3. 人在场核验新的joint执行器/停止/home与场景，之后shadow，再受控模型执行。
   软件关节/TCP范围不是整臂碰撞规划；旧DP已成功不等于这条新链已现场通过。
4. 当前不做charger、RLT或泛化评估，也不上传原始图像、权重或凭据。
