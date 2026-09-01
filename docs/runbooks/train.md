# Train RoboTwin ACT

[简体中文](../zh-CN/runbooks/train.md)

Bootstrap the exact upstream tree, convert real episodes, then run the tracked
training wrapper:

```bash
scripts/bootstrap_robotwin.sh
ur5e-real process-act OUTPUT_RUN \
  --task pick_block_bowl --task-config simple --episodes 15
scripts/train_act.sh pick_block_bowl simple 15 0 0
```

The processed dataset and checkpoints live inside the ignored upstream runtime
tree. The adapter writes the required `SIM_TASK_CONFIGS.json` entry there; no
generated data or modification of RoboTwin is committed to this repository.

A production DP training wrapper is not implemented yet. Do not reuse the ACT
command and present it as DP; first complete the vector/Zarr adapter and offline
round-trip gate in the roadmap.
