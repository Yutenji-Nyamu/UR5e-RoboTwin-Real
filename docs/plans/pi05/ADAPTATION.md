# 原生 π0.5 与当前真机代码的适配审查

2026-09-07；实现基线：本仓库 `aeadb3eb7cada7abaa6509ac2f4a6c719113bd0f`，RoboTwin `210720340637cb4619283b295dde4cdd807c9e66`。源码检查不是运行验证。

当前决策是 **native π0.5 + joint-space + 先补采双记录**。不在此展开 RLT。总体方案见 [README](README.md)。

## 1. “参考 DP”具体复用到哪里

| 已有文件 | 核验事实 | π0.5/joint 适配 |
| --- | --- | --- |
| [collection/session.py](../../../src/ur5e_real/collection/session.py) | freedrive、相机、夹爪事件、结果 manifest | 保留流程；同时写入 q/TCP，扩展 schema；不破坏 review/abort |
| [hardware/rtde.py](../../../src/ur5e_real/hardware/rtde.py) | collector 只订阅 timestamp、actual_TCP_pose，receive 返回 `(time,pose)` | 新完整 RobotState 接口/兼容 facade，旧调用者不被返回值变更打断 |
| [data/schema.py](../../../src/ur5e_real/data/schema.py) | 旧 14D 是 `[tcp6,0,tcp6,g]`，TCP rotvec 连续化 | 新 q 字段与布局版本；不能对 q 应用 rotvec 处理 |
| [data/convert_hdf5.py](../../../src/ur5e_real/data/convert_hdf5.py) | 图像与状态按记录时间最近邻；HDF5 的 left/right_arm 均填 TCP | 双状态保留真实字段；表示选择必须显式；旧导出不改解释 |
| [DP/process_data.py](../../../src/ur5e_real/adapters/robotwin_dp/process_data.py) | `obs[t]` 和 `state[t]` 对 `action[t]=state[t+1]`；可裁首尾 | 复用 shift/边界/溯源思路；新导出采用 joint，并加事件保护尾段 |
| [DP/train_dp.py](../../../src/ur5e_real/adapters/robotwin_dp/train_dp.py) | 本仓库准备参数，调用原生 DP trainer；模型算法未重写 | π0.5 同样用本地入口构造原生 TrainConfig 调用原生 train.main |
| [DP/infer_dp.py](../../../src/ur5e_real/adapters/robotwin_dp/infer_dp.py) | 原生 DP + RealObservationEncoder + offline/shadow/execute + chunk/gripper | 保留工作流；加载器/观测字典换 π0.5；executor 选 joint |
| [control/chunk.py](../../../src/ur5e_real/control/chunk.py) | TCP 限速/500 Hz 插值；返回限幅 waypoint；回调在策略边界 | 复用时钟结构；新 joint 插值、每轴速度/步幅限制、取消与执行数量 |
| [control/servoj.py](../../../src/ur5e_real/control/servoj.py) | prime→start→mode2/runtime2；目标变量、初始化和 getter 都是 TCP | joint target 必须从当前 q 初始化，不能沿用 TCP 初值；同步提供 q/TCP |
| [robot-side 脚本](../../../robot_programs/servoj_control_loop.script) | 输入六维 pose→`get_inverse_kin(...,get_actual_joint_positions())`→servoj | 增加独立 joint 脚本/明确模式，直接解释六维 q；旧脚本保持 |
| [控制 RTDE XML](../../../robot_programs/control_loop_configuration.xml) | 已订阅 actual_q、actual_TCP_force/pose/speed 等；没有 timestamp | 新完整控制状态补时间；“XML订阅”与“Python缓存/落盘”分开 |
| [control/prepare.py](../../../src/ur5e_real/control/prepare.py) | home 只比较 TCP 并 moveL | joint 基线需要确定的 home q，不只验证末端位姿 |

这说明不必重新造硬件栈；但用户改选 joint 后，目标解释与初始化不能照搬 TCP runner。新增 joint 路径优先保持旧 TCP/DP 的行为不变，再考虑重构共用基类。

### 控制默认值的区分

当前 DP 明确传入 `max_linear_velocity=0.40 m/s`，而通用 `ChunkStreamConfig` 缺省是 `0.05 m/s`；不能只读底层默认就断言 DP 正在用 0.05。两者都不能直接成为 joint 的 rad/s 参数。

joint 的最小防混用测试：相同 shape=6 的 q 与 TCP 也必须因元数据不同而拒绝；关节模式启动时写入的是 measured q，不是 0、不是真实 TCP；只停止当前控制进程，不改变旧重播默认后端。

## 2. RoboTwin `policy/pi05` 原生部分

官方提供 π0.5 数据、训练与评估入口；此处对照的是当前锁定版本，不假设官方网页与每个辅助文件完全一致。[官方接入说明](https://robotwin-platform.github.io/doc/usage/Pi05.html)

以下本地链接指向可由锁文件重建的 `.third_party`；永久目录入口：[锁定版本源码](https://github.com/RoboTwin-Platform/RoboTwin/tree/210720340637cb4619283b295dde4cdd807c9e66/policy/pi05)。

| 原生文件 | 核验结果 | 计划 |
| --- | --- | --- |
| [training/config.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/training/config.py) | `pi05_aloha_full_base`：pi05=True、batch64、20k steps、H50/D32 | 派生 UR5 配置，不直接跑原默认规模 |
| [aloha_policy.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/policies/aloha_policy.py) | `adapt_to_pi` 控制轴符号和夹爪标定；缺图有 mask；输出前14维 | 对 UR5 关闭标定；缺失左 wrist 用原生 mask；14D 兼容可用 |
| [pi0_config.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/models/pi0_config.py) | π0.5 的 state 离散化进入 language tokens；max_token_len 默认200 | 保留其 tokenizer/状态机制，不套 DP 的连续 history encoder |
| [transforms.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/transforms.py) | DeltaActions/AbsoluteActions、quantile normalize、resize/pad | 保留归一化前 joint-delta 与一次 inverse；输出给 executor 是绝对 q |
| [models/pi0.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/models/pi0.py) | 共享 VLM/action expert；flow matching；采样默认10步 | 不改模型算法；输入输出的语义由 adapter 保证 |
| [scripts/train.py](../../../.third_party/RoboTwin/policy/pi05/scripts/train.py) | 冻结参数转 bf16；只给 `trainable_filter` 建 optimizer 和梯度 | 可实现 expert-only；验证实际参数树、一步 diff 和 optimizer 集合 |
| [policy_config.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/policies/policy_config.py) | config + checkpoint + assets 创建可 infer 的 policy；也识别 PyTorch | 本轮优先同一 JAX 原生链，不把“有 PyTorch 加载”当存在完整原生 PyTorch trainer |
| [serving/websocket_policy_server.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/serving/websocket_policy_server.py) | 原生 policy server | 同机隔离模型/硬件环境；本地绑定，元数据包含动作空间 |

## 3. action-expert-only 冻结：原生已具备机制

`TrainConfig.freeze_filter` 决定冻结集合，`trainable_filter = Param AND NOT freeze_filter`。trainer 用该集合做 `opt_state` 与 `DiffState`；因此不是只有 full/LoRA 二选一。

拟冻结：`PaliGemma/img/**` 与 language backbone；拟训练：action expert、`action_in_proj`、`action_out_proj`、π0.5 的 `time_mlp_in/out`。原生 Gemma 命名用无后缀作为 backbone、`_1` 作为 action expert，见 [gemma.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/models/gemma.py)。但须在实际 NNX tree 上核验路径与参数量，不靠宽泛 regex 盲冻。

验收必须同时满足：冻结参数在一步训练前后不变、expert/projection 至少有参数更新、optimizer 仅含预期 trainable 集合；EMA=None 时 checkpoint 可重载并推理。所有基础权重必须完整匹配预期 shape，不使用随机初始化的 expert 冒充 π0.5 base。

缺失相机置零并 mask 不一定减少到两份视觉计算：该锁定版 embed_prefix 仍遍历并编码每个 image entry。显存实测按真实实现计算，不机械乘 2/3。

## 4. 薄辅助脚本的具体问题

### 4.1 数据转换：保留原理，调整字段和频率

[scripts/process_data.py](../../../.third_party/RoboTwin/policy/pi05/scripts/process_data.py) 与现有 DP 一样，把当前 state/image 和下一帧 state 配成 action；不能已经 shift 的数据再 shift 一次。它还要求 per-episode instructions，现有真机 HDF5 没有这套目录，需要生成固定 task/prompt 或本地转换入口直接提供。

[convert_aloha_data_to_lerobot_robotwin.py](../../../.third_party/RoboTwin/policy/pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py) 存在这些明确假设：

- `fps=50`、14 个 ALOHA motor 名、3 张图；真机改成 10 Hz、明确 UR5 兼容字段名。
- OpenCV 解码为 BGR；写入 LeRobot 前明确转 RGB，与 live 保持一致。
- 目标 repo/cache 目录存在时 `shutil.rmtree`；本地入口应改为不可变新版本/已存在时报错，不调用这个覆盖分支。
- 默认 video 模式和多 writer；首版可用 image 模式和少量 writer，减少非必要编解码依赖。

原始双记录与导出分离；joint/TCP 选择在 exporter。建议新原始 schema 明确保留两种真实字段，adapter 最后才生成 14D 的 `joint_action` 兼容视图。

### 4.2 norm stats 脚本不是当前版本可原样执行的接口

[scripts/compute_norm_stats.py](../../../.third_party/RoboTwin/policy/pi05/scripts/compute_norm_stats.py) 调用 `_data_loader.create_dataset`，但锁定 [data_loader.py](../../../.third_party/RoboTwin/policy/pi05/src/openpi/training/data_loader.py) 提供的是 `create_torch_dataset(data_config,action_horizon,model_config)`，未定义前者。

此外该脚本 `batch[key][0]` 仅取第一个样本；实际 loader yield 的是普通 batched tensor，而非它所暗含的额外设备维度。其 `num_batches=num_frames` 又把帧数作为批次数，可能重复遍历。下一轮修本地辅助入口：完整覆盖选择的样本、按真实 batch/action-horizon 展平、处理尾 batch、报告样本数量并保存 q01/q99/mean/std；使用原生 data transforms 与 RunningStats，不另造归一化算法。

这是静态接口/索引审查发现，尚未运行该脚本触发报错；需加小型 synthetic dataset 测试后再计算真机 stats。

### 4.3 checkpoint 路径与 timing

锁定 [pi_model.py](../../../.third_party/RoboTwin/policy/pi05/pi_model.py) 中 checkpoint 路径仍为 `policy/pi0/checkpoints`。本地 adapter 显式传入完整路径及 config，避免继续依赖这个硬编码。

[Policy.infer](../../../.third_party/RoboTwin/policy/pi05/src/openpi/policies/policy.py) 的计时在 JAX 输出物化前停止；单看其内部 infer_ms 可能漏算异步设备执行。本地以“请求开始→动作转成 host 数组/响应接收完成”计端到端时延，先热身再记录 p50/p95。

### 4.4 环境和补丁管理

[pyproject.toml](../../../.third_party/RoboTwin/policy/pi05/pyproject.toml) 要求 Python>=3.11，含 JAX0.5.0/Flax0.10.2/Orbax0.11.1，以及锁定 LeRobot commit `a445d9c9da6bea99a8972daa4fe1fdd053d711d2`；现有硬件/DP 文档是 Python3.10。独立模型环境，不原地升级正在工作的环境。实际依赖冲突先按导入路径定位；不因真机不跑仿真就断言 vendor 所有仿真相关依赖都可随意删除。

本地配置可构造 `TrainConfig` 后调用原生 `train.main(config)`；compute/serve 同样复用原生模型与变换。若需要最小 vendor 修补，沿用 [bootstrap/patch 机制](../../../scripts/bootstrap_robotwin.sh)，并增强对新增补丁的识别；不要修改 ignored checkout 后忘记托管，也不要运行 bootstrap 覆盖未知改动。

## 5. joint 新路径不可省略的检查

- `actual_q` 是观测，不是 `target_q`；freedrive 示教首先用实际状态的下一步构造标签。以后有遥操作命令流时另存 desired action，不混成一个字段。
- 不对六个关节一律做 `[-π,π]` 取模；保留示教分支，检查异常跳变/物理范围。新 home q 和模型 metadata 保持一致。
- 不使用 `nearest_rotation_vector` 或 TCP 欧氏速度裁剪来处理 q；新增 joint 专用逻辑。
- 原脚本 `mode<3`，3 代表 stop；不能草率用 mode4 代表 joint，导致循环退出。建议独立 joint script，仍沿用0/2/3握手，配置层锁定解释。
- 端点 q 可达不代表中途整臂不碰撞；在固定 demo 范围内先用记录动作受控验证，核对 joint 跟踪误差与 TCP 轨迹，之后才连接模型输出。
- 新字段、joint 控制、数据导出、训练配置分别测试，不为顺带支持 joint 改坏旧 TCP DP/重播。

完成这些以后，剩余工作是按阶段实现与测量，不需要用户重新选择后端、模型家族或 RL 算法。
