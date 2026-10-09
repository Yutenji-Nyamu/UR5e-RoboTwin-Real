# GeoWAM 真机数据与训练

本分支 `codex/geowam-real-training` 记录深圳 2 机 GPU 6、7 的实现。主目录 `/data/chenyiteng/projects/geowam-real`。

- [本轮完成情况](11_本轮完成情况.md)
- [当前状态](00_STATUS.md)
- [数据处理](01_数据处理规划.md)
- [初始化与训练](02_初始化与训练规划.md)
- [交接与运行入口](03_重启交接.md)
- [来源](04_来源与待落实项.md)
- [首段处理验收](05_首段真机预处理验收.md)

训练选择固定为 72 条实测 RGB-D，五任务、单臂 UR5e、6 关节＋1 DOF gripper、10 Hz、H32。桌布、彩色光照及缺深度示范已分类排除。代码/配置在本仓库，原始图像、模型、缓存及运行输出在主目录对应目录，保持 Git 轻量。Overleaf 本轮修改为零。

流水线：`pipeline_common.py` 建立动作/时间契约 → `process_geometry.py` / `repair_masks.py` 生成并复查几何 → `cache_latents.py` 编码 → `finalize_cache.py` 验收 → `run_training.py` 两卡训练。入口均位于 `scripts/geowam/`。实时状态在 `runs/`。

## 夜间实现

- [自主决策](06_OVERNIGHT_DECISIONS.md)
- [数据 QA](07_DATA_QA.md)
- [训练与拟合](08_TRAINING.md)

- [模块初始化](09_INITIALIZATION.md)

- [运行、预测图与交接](10_运行与交接.md)
