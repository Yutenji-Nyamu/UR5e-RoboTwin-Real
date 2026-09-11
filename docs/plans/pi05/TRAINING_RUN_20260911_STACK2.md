# π0.5 叠两个方块0911：五条示教 SFT

2026-09-11。用户已授权：新任务 `stack_blocks_two_0911`，沿用此前成功配方，克制裁剪、启动训练并持续监控。
当前状态：5条数据审计、双相机关键帧复核、504样本不可变导出及原生加载检查通过，待启动同配方SFT。

## 数据

- 仅选择本任务5条 `v3 / completed / success`，不混入叠三块、pour_beans或旧cube数据。
- 双相机关键帧均为抓黄块、放到红块上；使用 prompt：`Stack one block on top of the other.`
- [选择配置](../../../configs/pi05_stack_blocks_two_0911_joint_5.json)；[完整来源与裁剪审计](STACK2_0911_JOINT5_DATA_AUDIT.json)。
- 沿用关节运动阈值0.005rad、首尾3帧余量，只裁连续首尾静止；不裁中间、不平滑、不重排动作。
  每条保留完整一次close→open，最后训练观测距open至少1秒，再留0.1秒提供next-state标签。
- 10Hz，实测q[t]→q[t+1]，夹爪零阶保持、图像最近邻；14D兼容表示、原生delta/quantile/H50/padding不变。
  原始开头的0.2秒图像间隔均在裁掉部分，保留段没有复用图像。
- 不修改原始CSV、图片、成功标签；归一化统计仅从本次新数据生成。

| run ID | 原始图像对 | 裁头(s) | 裁尾(s) | 训练样本 | 最后观测距open(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260911_170047 | 132 | 1.9 | 1.5 | 98 | 1.0 |
| 20260911_170120 | 138 | 1.5 | 1.4 | 109 | 1.0 |
| 20260911_170221 | 127 | 1.2 | 2.0 | 95 | 1.0 |
| 20260911_170321 | 129 | 1.7 | 1.5 | 97 | 1.0 |
| 20260911_170406 | 140 | 1.6 | 1.9 | 105 | 1.0 |
| 合计 | 666 | 7.9 | 8.3 | 504 | |

本地复核图：`outputs/pi05_stack2_0911_joint5_review.png`，不上传原始图像到Git。
不可变数据集：`/data/robotics/ur5e-real/pi05/lerobot/ur5e/stack_blocks_two_0911_joint_5_v20260911`。
导出约365MiB；原生LeRobot加载、15个观测的训练/RPC图像和状态一致、quantile/delta/inverse往返检查通过，
模型动作50×32，缺失左腕mask为false。本轮复用已有训练代码，没有新增软件改动。
示教关节速度峰值约0.994rad/s（wrist_1）；本轮不修改轨迹时间或真机控制参数。

## 固定训练配方

沿用 [已成功的cube配方](TRAINING_RUN_20260909.md) 和 [上一轮叠三块SFT](TRAINING_RUN_20260911_STACK3.md)，
仅替换数据、任务prompt、归一化统计及实验名；从官方完整 `pi05_base` 重新SFT，不续训旧任务权重。

- 实验：`stack2_0911_joint5_sft_20260911_01`；1000次optimizer更新，LR日程仍3000步。
- batch8、workers2、LR2.5e-5、warmup100、seed42；8000样本次约为本数据15.9遍。
- 冻结视觉/语言骨干，仅训练action expert及action/time投影；EMA、W&B、图像增强关闭，flow噪声保留。
- 每500步及最终保存和评估，每条固定3个训练观测；评估去噪10步，统计剔除末尾padding后的q6/夹爪误差。
- 资源每2秒采样、每30秒控制台摘要；逐步记录loss、梯度、学习率和耗时。
- 结束检查冻结参数不变、可训练参数改变、参数及optimizer保存重载；1000步不自动延长。

训练命令（由独立后台服务调用，不要重复运行）：

```bash
.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 train \
  --dataset /data/robotics/ur5e-real/pi05/lerobot/ur5e/stack_blocks_two_0911_joint_5_v20260911 \
  --exp-name stack2_0911_joint5_sft_20260911_01 \
  --steps 1000 --schedule-steps 3000 --batch-size 8 --num-workers 2 \
  --learning-rate 2.5e-5 --warmup-steps 100 \
  --save-interval 500 --eval-interval 500 --eval-points 3 --no-image-augmentation \
  --monitor-interval 2 --monitor-console-interval 30
```

## 运行、监控与交接

- 启动前上一轮叠三块已正常退出（MainPID0、退出码0），1000步四项冻结/重载审计通过。
  A6000当前无计算进程，占用610MiB、空闲47,923MiB；数据盘约3.0TiB、checkpoint盘约387GiB可用。
- 新后台服务预定：`ur5e-pi05-stack2-0911-joint5-sft-20260911-01.service`。
- 控制台：`logs/pi05_stack2_0911_joint5_sft_20260911_01.console.log`。
- 结构化日志、启动时间、首批optimizer更新及回访状态：启动后补记。
- 完成须同时确认 `result.json`、四项审计和服务正常退出；`active/exited` 是已结束，不是还在训练。
- 预计约30分钟量级，包含编译、保存、评估、重载，不以optimizer步数单独判断是否卡住。
- 计划恢复当前任务的定时回访，每5分钟检查，正常推进不重复汇报；首次500步评估、完成、失败或需行动时报告。
  监控不自动改参数、重启、延长训练或操作机器人；完成/失败后暂停。
- 不覆盖cube/叠三块数据及checkpoint；本轮不更改推理逻辑，不启动真机或RLT。
  开始时已有 `infer.py`、`tests/test_pi05.py` 的未提交改动，保留，不混入本次数据与训练记录提交。
- 同日已有叠三块实验，后续模型引用使用完整 `stack2_0911_joint5_sft_20260911_01:1000`，
  不用可能歧义的 `20260911_01:1000`；训练和审计完成前不将它当可用模型。

只读状态：

```bash
systemctl --user show ur5e-pi05-stack2-0911-joint5-sft-20260911-01.service \
  --property=ActiveState,SubState,MainPID,ExecMainStatus,Result
tail -n 15 logs/pi05_stack2_0911_joint5_sft_20260911_01.console.log
```

## 结果

待训练。500/1000步评估是训练集拟合诊断，不等于真机成功率。
