# 当前状态（深圳 2）

工作区 `/data/chenyiteng/projects/geowam-real`；分支 `codex/geowam-real-training`；物理 GPU 6、7。

- 数据与模型全部下载校验完成。训练固定 72 条实测 RGB-D，共 2,405 个窗口；五任务，单臂六关节＋1 DOF gripper。
- 原始图像、SAM 掩码、RAFT、UVD 与 VAE 已完成多轮复查；12 条记录共 44 个提示帧针对性修复。全部缓存通过最终验收。
- 两卡 batch 1/2 smoke、完整检查点恢复、五任务 16 轮原生采样、当前输入因果编码检查均通过。
- 正式配置为 FSDP2 block、BF16、每卡 batch 2、累积 1、30,000 步；每小时保存检查点并生成五任务拟合曲线。当前开始启动，实际状态见下列文件。
- RLT 维持 6/7 卡交接；本训练及其续训优先。4/5 卡的其他工作继续运行。

实时入口：`runs/training_status.json`、`runs/training_health.json`、`runs/train_rgbd72_v1/train_log.jsonl`。健康报告由 `scripts/geowam/monitor_training.py` 生成。

恢复入口：先核对上述状态与本任务进程，完整检查点位于 `runs/train_rgbd72_v1/checkpoints/step_*/complete.json`；`run_training.py train` 会从最近完整检查点恢复。

[文档索引](README.md)。Overleaf 本轮新增 0、修改 0、删除 0。
