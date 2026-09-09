# π0.5 首轮正式 SFT：2026-09-09

**状态：1000步正式SFT及最终审计已完成，2026-09-09 17:28:41 CST正常退出；监控已暂停。**
机器可读汇总见 [训练结果](TRAINING_RESULT_20260909.json)。未进行真机执行。

用户已授权启动正式训练并持续监控。本次已运行到 **1000 steps**，不是3000步；
3000是固定学习率日程，便于之后经确认续训。无真机连接、无自动重启或自动续训。

## 运行定位

- 实验：`joint5_sft_20260909_01`，全新实验，不沿用3步开发checkpoint。
- 启动：2026-09-09 16:57:31 CST（08:57:31 UTC）。
- 训练代码：`c7e95a791fcb98b5f5be6c15f27b51bc4dba9dbb`；后续文档提交不改变此运行代码。
- 原生 RoboTwin：`210720340637cb4619283b295dde4cdd807c9e66`，使用既有3个托管兼容补丁。
- Python：项目 `.venv/pi05/bin/python`；官方完整 `pi05_base` 命中本机已校验缓存。
- 后台服务：`ur5e-pi05-joint5-sft-20260909-01.service`（systemd user service），初始主PID `74273`。
  `Restart=no`；独立于聊天终端。机器重启不自动重开训练。
- 控制台：`logs/pi05_joint5_sft_20260909_01.console.log`。
- 结构化日志：`logs/pi05/joint5_sft_20260909_01/20260909T085731277973Z/`。
- checkpoint目录：`checkpoints/pi05/pi05_ur5e_joint_action_expert/joint5_sft_20260909_01/`。
  `500`、`1000`均已完整保存，各约8.8GiB；`1000`额外通过最终参数/optimizer重载审计。

## 固定配置

| 项目 | 本次值 |
| --- | --- |
| 数据集 | `ur5e/pick_place_cube_joint_5_v20260908`，5条、414个窗口，全部用于拟合 |
| 本地数据 | `/data/robotics/ur5e-real/pi05/lerobot/ur5e/pick_place_cube_joint_5_v20260908` |
| batch / loader workers | 8 / 2 |
| optimizer更新预算 / LR日程 | 1000 / 3000 steps |
| LR / warmup / seed | peak 2.5e-5 / 100 steps / 42 |
| 可训练范围 | action expert和action/time投影；430,098,464个参数 |
| 冻结范围 | vision/language backbone；2,923,335,408个参数 |
| 时序与表示 | 10Hz、H50；外部绝对joint，内部原生delta；物理q6＋夹爪 |
| 图像增强 / EMA / W&B | 关闭 / 关闭 / 关闭；保留原生flow噪声和loss |
| 保存与评估 | 每500步及最终；每条示教固定3个观测，排除padding，分别看物理q6/夹爪和前6步 |
| 资源采样 / 控制台摘要 | 每2秒 / 每30秒；逐步记录loss、梯度、LR、耗时和JAX显存 |

启动参数与 [操作说明](USAGE.md) 的正式SFT示例相同，实验名为上述唯一名称；
另显式使用 `--eval-points 3 --monitor-interval 2 --monitor-console-interval 30`。
完整不可变recipe和本次invocation已保存到结构化日志目录。

## 持续监控与停止条件

- 原每5分钟回访；automation ID `0-5`，名称“π0.5 首轮训练监控”。17:28完成后已设为`PAUSED`，不重复通知。
- 没有需行动变化时保持安静；首次500步评估、完成、失败或需处理问题时通知。
- 分辨加载/JIT编译/训练/保存/评估/最终重载阶段，不能将无新optimizer step一概当作卡住。
- 检查loss/梯度有限、资源采样错误、GPU余量和系统available内存；失败时保留证据，不改超参重开。
- 同时检查长跑温度；本机驱动报告Max Operating 93°C、Slowdown 95°C。
  若温度接近/超过93°C或持续热降频，核对驱动实际状态并报告；不自行改功率、风扇或驱动设置。
- 1000步后还要等冻结哈希审计、模型参数与optimizer重载，以及进程正常结束。
  `RemainAfterExit=yes`使正常完成后服务可能仍显示active/exited；此时MainPID为0，不是仍在训练。
- 完成或失败后更新本文件，提交推送相关文档，然后暂停本监控，不归档任务。
- 本机监控依赖电脑和应用保持运行；后台训练本身不依赖聊天终端。
  监控方式按 OpenAI Docs 核对的[官方定时任务说明](https://learn.chatgpt.com/docs/automations?surface=app)设置。

只读状态命令：

```bash
systemctl --user show ur5e-pi05-joint5-sft-20260909-01.service \
  --property=ActiveState,SubState,MainPID,ExecMainStatus,Result
tail -n 20 logs/pi05_joint5_sft_20260909_01.console.log
```

需要人工中止时仅停止本服务，不影响其他训练：

```bash
systemctl --user stop ur5e-pi05-joint5-sft-20260909-01.service
```

中止不会额外保存尚未到保存点的更新；已有完成checkpoint保留，不自动续训。

## 已观察记录

- 启动前：A6000可用显存47,968MiB，无其他GPU计算进程；系统available约115GiB；
  checkpoint所在盘可用约428GiB。以上是当时快照，不是持续余量承诺。
- 16:58：完整基础权重加载、初始参数审计完成；冻结和可训练组哈希与此前真实base短测一致。
- 16:59:08首个optimizer step完成（首步编译＋计算27.30秒）；17:01:03已到72步，loss与梯度均有限，
  训练＋取数约1.6秒/步，显存约16.9GiB。loss从首步0.203降到第72步0.044，但单步loss有噪声，未据此宣称拟合成功。
- 17:01驱动检查：GPU 87°C，当次SW/HW Thermal Slowdown均Not Active，功耗上限300W；继续观察温度。
- 17:13：第500步checkpoint保存和离线评估完成，之后自动继续本次1000步预算。
  评估15个固定训练观测、464个有效目标，排除286个末尾padding目标；并非全数据或闭环评估。
  有效q6 MAE **0.013879rad（0.795°）**，前6步q6 MAE **0.007137rad（0.409°）**；
  夹爪准确率92.89%，前6步在这些抽样点上100%。后者不能代替开合切换时刻或真机释放验证。
  评估耗时9.06秒；manifest状态`intermediate_offline_only`，尚未进行最终冻结和重载审计。
- 17:15:33 loss快照到592步：前50步均值0.163014，最近50步0.025443，下降84.39%；
  301～350步均值0.029387，451～500步0.026792，后期仍有小幅改善与随机波动。
  结论是明显收敛趋势、进入缓慢下降阶段，不能仅凭含padding/兼容维度的原生loss认定充分拟合。
  第500步结果与曲线已在用户询问loss的回复中报告，监控不重复通知该节点。
- 本地曲线：`outputs/pi05_loss_20260909_snapshot_01.png`，同名JSON记录窗口统计与快照时间；
  文件为CPU绘图产物，不加载另一份模型、不修改正在运行的训练。
- 17:26:41完成第1000步；保存12.78秒、固定训练观测评估2.81秒，随后进行完整参数/optimizer哈希及重载。
- 17:28:41服务正常退出，`Result=success / ExecMainStatus=0 / MainPID=0 / SubState=exited`；
  `result.json`与checkpoint的`ur5e_verification.json`一致（前者另含checkpoint路径）。
  冻结参数哈希不变、可训练参数哈希改变、参数与optimizer重载全部通过。
  后续GPU无计算进程，整卡显存已回落到约578MiB。

## 最终结果与下一步

全程31分10秒（含启动、编译、保存、评估、哈希与重载），模型运行器计时1858.81秒；
999个非首步的平均样本吞吐4.919样本/秒、单步中位1.621秒，吞吐不含保存/评估/审计。
共8000个训练样本次，约19.3倍414样本数据量；不是8000条独立示教。

| 固定训练观测评估 | 第500步 | 第1000步 | 解释 |
| --- | ---: | ---: | --- |
| 有效全H q6 MAE | 0.013879rad（0.795°） | 0.009411rad（0.539°） | 下降32.19% |
| 前6步q6 MAE | 0.007137rad（0.409°） | 0.003976rad（0.228°） | 下降44.29% |
| 有效全H夹爪准确率 | 92.89% | 92.67% | 431/464 → 430/464，基本持平，无改善结论 |
| 前6步夹爪准确率 | 100% | 100% | 仅65个有效prefix目标，不能代替切换时刻验证 |

两次使用同样15个训练观测和固定采样噪声，464个有效目标、286个padding排除；
首版暂未运行全轨迹预测检查、独立进程的本checkpoint服务推理或真机shadow/execute。
状态明确为`SFT_not_physical_validation`，不是“真机demo已跑通”。

- 最后50步loss均值0.023998（前50步0.163014，下降85.28%）；最后100步0.023207。
  末尾相较925步时的滚动均值0.02164有小幅回升，属于本次观察到的波动，不能写成单调下降。
  后期接近低位平台，但500→1000的实际关节误差仍有改善；不只按loss作续训判断。
- 最终曲线为本地artifact `outputs/pi05_loss_20260909_final_1000.png`，同名JSON含完整50步分段统计。
- 895次资源采样：整卡已用峰值16.99GiB、可用最低30.40GiB；JAX活跃分配峰值14.39GiB；
  系统available最低84.45GiB、主进程RSS峰值31.15GiB，资源采样错误为0。GPU温度采样峰值88°C。
  NVML采样峰值可能漏瞬时峰值；不能混同JAX allocator统计。
- 完成后通过官方定时任务工具暂停automation `0-5`（按OpenAI Docs流程），未归档任务、未删除checkpoint。
- 建议下一步先用1000步checkpoint做五条示教的全轨迹离线预测检查，重点看夹爪切换、动作连续性和约束；
  再进入现场shadow/受控执行。此为建议，**本轮未启动这些动作，也没有自动续到3000步**。

### 重画loss快照

使用已有含matplotlib的硬件/数据环境，**不向π0.5训练环境安装依赖**。每次选择新的输出文件名：

```bash
MPLCONFIGDIR=/tmp/ur5e-pi05-matplotlib \
  /home/zhangw/anaconda3/envs/RoboTwinSimReal/bin/python scripts/plot_pi05_loss.py \
  --metrics logs/pi05/joint5_sft_20260909_01/20260909T085731277973Z/metrics.jsonl \
  --output outputs/pi05_loss_NEW_SNAPSHOT.png
```

脚本只读取已写完整的JSONL行，绘制原始loss和50步滑动均值，以及后期放大图；
输出不可覆盖，保存PNG与统计JSON。两幅图纵轴不同；整体图不截掉初期高loss。

## 500 / 1000步交接检查

1. 读取 `evaluation_500.json` / `evaluation_1000.json`，记录有效joint MAE、夹爪准确率和前6步误差，
   不只看包含末尾补齐的总loss；这些是训练集teacher-forced预测，不是真机闭环成功率。
2. 最终读取 `result.json` 与服务退出状态；要求step1000、frozen_unchanged、trainable_changed、
   checkpoint_reload、optimizer_reload均通过，并核对 `ur5e_verification.json`。
3. 汇总整段耗时、资源峰值/余量和后期loss曲线；保存和编译开销不能混入短测吞吐作等同比较。
4. 本次1000步结束后先报告拟合结果和下一步建议；不自行启动3000步或真机执行。
