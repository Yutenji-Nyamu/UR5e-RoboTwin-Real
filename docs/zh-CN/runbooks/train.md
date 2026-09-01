# 训练 RoboTwin ACT

[English](../../runbooks/train.md)

先bootstrap准确上游版本，再转换真实episode并运行受跟踪的训练包装脚本：

```bash
scripts/bootstrap_robotwin.sh
ur5e-real process-act OUTPUT_RUN \
  --task pick_block_bowl --task-config simple --episodes 15
scripts/train_act.sh pick_block_bowl simple 15 0 0
```

处理后数据和checkpoint位于被忽略的上游运行树中。适配器会写入所需的
`SIM_TASK_CONFIGS.json` 项；生成数据和RoboTwin运行时改动都不提交到本仓库。

DP尚未提供正式训练包装命令；在数据适配与离线往返测试完成前，不复制ACT命令
假装DP流程已经可用。
