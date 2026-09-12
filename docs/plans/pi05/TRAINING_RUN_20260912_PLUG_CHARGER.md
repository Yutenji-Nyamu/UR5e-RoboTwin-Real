# π0.5 插插座：五条数据，1500步SFT

2026-09-12。用户授权：整理今天新采的5条插插座数据，克制处理首尾，沿用此前参数训练π0.5，本次1500步。
当前状态：1216样本已不可变导出（约738MiB），原生检查通过；15:06:37 CST已启动1500步训练，
15:08首批真实optimizer更新正常，当前任务5分钟回访已恢复。

## 数据与唯一适配

任务 `plug_charger`；5条均为schema3、completed、人工标记success，实测关节/TCP字段齐全。
双相机关键帧符合抓取充电器并插入插座的过程；prompt为 `Pick up the charger and plug it into the socket.`。

- [选择配置](../../../configs/pi05_plug_charger_joint_5_20260912.json)；[来源哈希和完整裁剪审计](PLUG_CHARGER_JOINT5_DATA_AUDIT_20260912.json)。
- 仍是0.005rad关节运动阈值、首尾3帧余量，只裁连续首尾静止；不删中间插入/调整过程，不平滑或重排动作。
- 10Hz、q[t]→q[t+1]，夹爪零阶保持、图像最近邻；H50、14D兼容、原生delta/quantile/padding不变。
- 每条原始事件为 `close→close→open→open`，实际二值状态只发生一次合/开。
  原导出器严格要求单次close/open，故为本清单显式开启 `allow_repeated_gripper_commands=true`：
  仅按状态变化检查周期数，全部4个原事件及时间保留到审计；仍逐个验证时间，原始文件不改。
  旧配置默认不接受重复，不放宽错序或多次真实抓放的校验。
- **训练标签仍是开/合状态，不能表达第二次按键或额外夹紧/张开量**；这不是事件回放方案。
  两次close发送相同底层串口命令，不能由此认定物理效果相同。本轮不修改夹爪控制或推理终止逻辑。
- 尾段从最后一次原始open命令起保留至少1秒训练观测，再加0.1秒next-state标签；
  不是从较早的二值open转换裁断。每条源记录均保留，不覆盖旧cube/叠块数据。

| run ID | 原始图像对 | 裁头(s) | 裁尾(s) | 训练样本 |
| --- | ---: | ---: | ---: | ---: |
| 20260912_144757 | 253 | 1.6 | 0.5 | 232 |
| 20260912_144852 | 225 | 1.3 | 2.3 | 189 |
| 20260912_144941 | 304 | 1.4 | 0.5 | 285 |
| 20260912_145043 | 253 | 1.4 | 0.9 | 230 |
| 20260912_145151 | 307 | 1.4 | 1.3 | 280 |
| 合计 | 1342 | 7.1 | 5.5 | 1216 |

各条最后训练观测距最后open为1秒（浮点误差约1e-13），保留段无复用图像。
本地复核图 `outputs/pi05_plug_charger_joint5_review_20260912_v2.png`，末次open展示按原始命令时刻定位。
新数据集：`/data/robotics/ur5e-real/pi05/lerobot/ur5e/plug_charger_joint_5_v20260912`。

## 训练参数

沿用 [上一轮配方](TRAINING_RUN_20260911_STACK2.md)，仅更换任务数据/prompt/归一化统计和实验名，
并按用户要求将停止步数由1000改为1500；从官方完整 `pi05_base` 重新SFT，不续训旧任务。

- 实验 `plug_charger_joint5_sft_20260912_01`；batch8、workers2、seed42。
- 1500次optimizer更新；学习率日程仍3000步、peak LR2.5e-5、warmup100。
- 冻结视觉/语言骨干，只训练action expert及action/time投影；EMA、W&B、图像增强关闭，flow噪声保留。
- 保存/评估点500、1000、1500，保留三个版本；固定每条3个训练观测，评估去噪10步。
  中间500/1000仅供评估或续训；1500结束后做完整冻结/参数/optimizer重载检查。
- 资源每2秒采样、每30秒摘要；逐步记录loss、梯度、LR、吞吐。
- 12000样本次约9.9遍本数据；预计约45分钟量级，不自动延长或重启。

训练命令（已由独立后台服务调用，不要重复启动）：

```bash
.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 train \
  --dataset /data/robotics/ur5e-real/pi05/lerobot/ur5e/plug_charger_joint_5_v20260912 \
  --exp-name plug_charger_joint5_sft_20260912_01 \
  --steps 1500 --schedule-steps 3000 --batch-size 8 --num-workers 2 \
  --learning-rate 2.5e-5 --warmup-steps 100 \
  --save-interval 500 --eval-interval 500 --eval-points 3 --no-image-augmentation \
  --monitor-interval 2 --monitor-console-interval 30
```

## 验证与运行记录

- 开始时A6000无计算进程，空闲47,951MiB；RAM available约116GiB，checkpoint盘369GiB、数据盘3.0TiB可用。
- 重复命令兼容及π0.5回归：53 passed、3 skipped；绘图预算1000/1500/历史默认三个测试通过；Ruff检查通过。
- Loss绘图工具现从每次运行的 `invocation.json` 读取总预算，避免将1500步误画成1000步。
- 原生LeRobot加载1216样本、15个观测的训练/RPC RGB与状态一致、quantile/delta/inverse往返检查通过；
  action_shape仍50×32，缺失左腕mask为false。未将导出未结束时提前检查产生的缺文件报错计作有效验证。
- 启动：2026-09-12 15:06:37 CST，代码/配置 `9b7b3b6`；
  服务 `ur5e-pi05-plug-charger-joint5-sft-20260912-01.service`，初始MainPID35915，
  `Restart=no`、`RemainAfterExit=yes`，独立于聊天终端运行。
- 控制台 `logs/pi05_plug_charger_joint5_sft_20260912_01.console.log`。
- 结构化日志 `logs/pi05/plug_charger_joint5_sft_20260912_01/20260912T070637666460Z/`。
- checkpoint目录 `checkpoints/pi05/pi05_ur5e_joint_action_expert/plug_charger_joint5_sft_20260912_01/`。
- 已直接比较不可变recipe，除contract/dataset_ready外与上一轮完全一致；invocation明确1500步，
  save/eval间隔500、eval_points3、resume=false；官方基础权重命中本地缓存。
- 15:08:02完成第1次真实更新；初始分组frozen 2,923,335,408、trainable 430,098,464与旧配方相同。
  15:08:04已到第2步，显存约16.93GiB、余量30.47GiB，资源采样无错误。
- 已复用当前任务5分钟回访 `0-5`，改名“π0.5 插插座1500步训练监控”，ACTIVE，
  按OpenAI Docs的[同会话定时任务](https://learn.chatgpt.com/docs/automations)流程管理；
  正常推进保持安静，500/1000评估、1500完成或异常时汇报。保持电脑与应用运行。
- 完成必须同时具备result.step=1500、四项冻结/重载检查true和正常退出；不能在1000步停止监控。
- 开始时已有 `infer.py`、`tests/test_pi05.py` 未提交改动，保留且不混入本次提交；原始数据、权重、大日志不进Git。
- 本轮没有机器人动作，没有启动RLT；训练集拟合不代表插插座真机已成功。

已报告节点：数据整理、启动；500/1000/1500步结果待回访。
