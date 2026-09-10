# π0.5 方块真机成功记录

**结论：原生 RoboTwin π0.5 → UR5e joint-space 抓放方块最小 demo 已跑通。**
现场操作者于 2026-09-10 明确反馈成功；本地另有两份 09-09 正常完成的 execute 日志。
两类证据分别记录，不据此声称“2/2 成功率”或已经验证泛化。

## 固定基线

| 项目 | 本次值 |
| --- | --- |
| 操作代码 | `5342f1f`（两条短命令交付）；日志没有逐次记录 Git SHA，不追认成日志自证 |
| 上游 | `RoboTwin@210720340637cb4619283b295dde4cdd807c9e66`，原生 JAX π0.5 |
| 训练 | `joint5_sft_20260909_01`，1000 steps，batch 8，5 条 joint 示教 / 414 transitions |
| checkpoint | `checkpoints/pi05/pi05_ur5e_joint_action_expert/joint5_sft_20260909_01/1000` |
| 数据 | `ur5e/pick_place_cube_joint_5_v20260908`；实际 q + TCP 双记录，训练使用 joint |
| 模型/执行 | H=50，K=20，10 Hz 目标，10 flow steps；500 Hz joint servoJ，限速 0.6 rad/s |
| 观测 | head + wrist RGB；缺失左腕 mask；q6 + 夹爪命令状态 |
| 停止 | 一次 close→open 后保持 1 秒；只是结束规则，不是视觉成功分类器 |

## 日志摘要

原始本地目录为 `logs/pi05_infer/<下表 ID>/`，没有上传到 Git。
`run.json` 均为 `execute=true`、checkpoint `20260909_01:1000`；
`result.json` 均为 `completed`、`automatic_success_judgement=false`。

| UTC 日志 ID | 中国时间开始 | chunk 数 | 实际目标点 | servo 设点数 | 计划执行时长合计 | RPC 范围 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260909T102055288817Z` | 09-09 18:20:55 | 5 | 98（末段18） | 4900 | 9.8 s | 0.245–0.266 s |
| `20260909T102230271522Z` | 09-09 18:22:30 | 4 | 70（末段10） | 3500 | 7.0 s | 0.251–0.278 s |

两次均无 retimed waypoint。计划时长不含加载、推理等待、开爪/结束等待；
servo 设点计数也不是独立逐点实测跟踪误差。RPC 共9次，不能当长期 p95 基准。

## 复用与后续

```bash
ur5e-pi05-infer-init
ur5e-pi05-infer 20260909_01:1000 --execute
```

以上针对已部署且保持同一标定/摆场的设备；新机器先读
[新机与检修手册](../../zh-CN/runbooks/new_machine.md)。

- 该 checkpoint 作为 RLT 的冻结参考基线保留；不覆盖、不为接 RLT 先重做 SFT。
- 训练产物中的 `SFT_not_physical_validation` 是训练时的审计标记，也是当前加载契约；
  不改写它，真机证据单独追加在本文件。
- 尚无自动 reward、逐回合人工标签入库、完整故障轨迹、成功率对照实验；
  这些属于 [cube RLT 后续规划](../pi05-rlt/CUBE_PLAN.md)，不是本次成功报告的前提。
- 原 TCP DP 与手动重播保持不变；不把 DP 的动作空间或 K=6 套给 π0.5。

相关：[训练结果](TRAINING_RUN_20260909.md) · [参数来源](INFERENCE_PARAMETERS.md) · [操作](USAGE.md)。
