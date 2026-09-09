# π0.5 首轮正式 SFT：2026-09-09

用户已授权启动正式训练并持续监控。本次只运行到 **1000 steps**，不是3000步；
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
  计划产生 `500`、`1000`；目录存在不等于保存/重载已通过。

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

- 本任务每5分钟回访；automation ID `0-5`，名称“π0.5 首轮训练监控”。
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
- 尚无500步checkpoint、完整拟合或真机成功证据；完整保存/重载结果必须等结束审计。

## 500 / 1000步交接检查

1. 读取 `evaluation_500.json` / `evaluation_1000.json`，记录有效joint MAE、夹爪准确率和前6步误差，
   不只看包含末尾补齐的总loss；这些是训练集teacher-forced预测，不是真机闭环成功率。
2. 最终读取 `result.json` 与服务退出状态；要求step1000、frozen_unchanged、trainable_changed、
   checkpoint_reload、optimizer_reload均通过，并核对 `ur5e_verification.json`。
3. 汇总整段耗时、资源峰值/余量和后期loss曲线；保存和编译开销不能混入短测吞吐作等同比较。
4. 本次1000步结束后先报告拟合结果和下一步建议；不自行启动3000步或真机执行。
