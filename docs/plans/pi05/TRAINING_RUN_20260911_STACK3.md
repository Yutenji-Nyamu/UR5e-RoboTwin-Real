# π0.5 叠三块：五条示教与复用 SFT 配方

2026-09-11。用户授权：选择刚采集的新任务成功数据，克制处理头尾静止，按上次成功 π0.5 的相同参数训练。
当前状态：数据审计、关键图像复核、不可变导出及原生加载检查通过，待启动后台训练。

## 数据与裁剪

- 任务 `stack_blocks_three`，选择5条 `v3 / completed / success`；同批 `20260911_154614` 为 failure，排除。
  更早的3条 `pour_beans` 失败记录属于另一任务，不纳入。
- 关键图像显示先叠两层，再叠第三层。前4条红在黄上、蓝在最上，第5条蓝在黄上、红在最上；
  都符合“叠三块”而非固定颜色顺序，使用同一 prompt：`Stack the three blocks into a tower.`
- 选择配置：[五条清单](../../../configs/pi05_stack_blocks_three_joint_5_20260911.json)；
  完整来源哈希、关节/TCP范围、事件时刻：[数据审计](STACK3_JOINT5_DATA_AUDIT_20260911.json)。
- 沿用原先的关节阈值 `0.005 rad`、首尾各3帧余量；只裁连续首尾，不裁中间停顿，不平滑/重排动作。
  末次 open 后保留至少1秒训练观测，额外0.1秒提供最后 next-state 标签。
  原始CSV、图片、人工成功标签及此前 cube 导出不修改。
- 10Hz、q[t]→q[t+1]，夹爪零阶保持，图像最近邻，native delta/quantile/H50/padding与上次相同。
  各条开头有0.2秒图像间隔，均在被裁去的起始段；保留段无复用图像。

| run ID | 原始图像对 | 裁头(s) | 裁尾(s) | 训练样本 | 最后观测距末次open(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260911_154322 | 215 | 1.6 | 0.8 | 191 | 1.0 |
| 20260911_154430 | 214 | 1.9 | 0.8 | 187 | 1.0 |
| 20260911_154520 | 224 | 1.5 | 1.0 | 199 | 1.0 |
| 20260911_154720 | 234 | 1.8 | 1.4 | 202 | 1.0 |
| 20260911_154813 | 269 | 2.0 | 3.5 | 214 | 1.0 |
| 合计 | 1156 | 8.8 | 7.5 | 993 | |

浮点误差约1e-13秒按1.0秒记录，不是额外裁剪。
本地关键帧图：`outputs/pi05_stack3_joint5_review_20260911.png`，未上传原始图像到Git。
新数据集：`/data/robotics/ur5e-real/pi05/lerobot/ur5e/stack_blocks_three_joint_5_v20260911`。

## 必要的软件适配与边界

旧导出器只接受一次 `close→open`，新示教每条为两次。
增加显式 `gripper_cycles=2`，校验全部事件的顺序/时间并记录到新动作契约；旧配置缺省仍为1次。
首尾裁剪方法、训练模型/loss/冻结范围不改。关键帧检查工具显示两次抓放。
回归：`139 passed, 3 skipped`，包含多周期事件、末次释放尾段、中间停顿保留和旧默认不放宽。
原生检查：993样本LeRobot加载、训练/RPC RGB与状态一致、quantile/delta/inverse往返均通过；
模型输入动作仍为50×32，缺失左腕mask保持false。数据导出约722MiB。

本轮只做数据与离线训练，没有连接/执行机器人。
现有真机执行器仍按第一次 close→open 后结束；这个旧结束条件不能用于新任务完整验收。
后续推理前需让它按新契约完成两次抓放，并检查中间释放后允许再次闭爪；本轮不改真机执行逻辑。
旧cube checkpoint与RLT运行不覆盖、不自动切换为新任务。

## 固定训练配方

逐项沿用 [上次正式训练](TRAINING_RUN_20260909.md)，只更换数据/任务prompt、归一化统计和独立实验名。

- 从完整官方 `pi05_base` 重新 SFT，不从旧cube的1000步权重续训。
- 实验：`stack3_joint5_sft_20260911_01`；1000 optimizer更新，固定3000步LR日程。
- batch=8，workers=2，peak LR=2.5e-5，warmup=100，seed=42。
- 冻结 vision/language backbone，仅 action expert 与 action/time投影可训练；EMA/W&B/图像增强关闭。
- 每500步及最终保存和固定训练观测评估（每条3点），结束做冻结/参数/optimizer重载审计。
- 资源每2秒采样、每30秒摘要；逐步loss/梯度/LR/耗时写JSONL。
- 8000样本次约等于新数据8.1遍；旧414样本约19.3遍。仍遵从1000更新，不自动加步。
- 推理的 K=50、去噪5/2不是本次训练参数；训练仍H50，诊断采样沿用10次去噪。

启动命令（待后台服务执行，不要同时重复启动）：

```bash
.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 train \
  --dataset /data/robotics/ur5e-real/pi05/lerobot/ur5e/stack_blocks_three_joint_5_v20260911 \
  --exp-name stack3_joint5_sft_20260911_01 \
  --steps 1000 --schedule-steps 3000 --batch-size 8 --num-workers 2 \
  --learning-rate 2.5e-5 --warmup-steps 100 \
  --save-interval 500 --eval-interval 500 --eval-points 3 --no-image-augmentation \
  --monitor-interval 2 --monitor-console-interval 30
```

预计沿用上次约31分钟量级；数据变长不会按示教时长倍增1000次固定batch更新的计算量。
实际耗时以新运行日志为准，启动编译/保存/审计均算在内。

## 运行与结果

启动前：A6000已用586MiB、空闲47,948MiB；系统available约116GiB，checkpoint盘可用405GiB。
后台服务、启动时间、首步与结果待启动后补充；没有把数据检查记成训练完成或真机成功。
