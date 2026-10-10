# 数据处理

按顺序阅读：

1. [原始数据与筛选](01_原始数据与筛选.md)：87 条怎样选成 72 条。
2. [时间、动作与窗口](02_时间动作与窗口.md)：10 Hz 数据怎样组成 32 步样本。
3. [SAM 掩码](03_SAM掩码.md)：提示、跟踪、角色与修复。
4. [RAFT 与 Track-UVD](04_RAFT与UVD.md)：二维光流怎样结合深度。
5. [编码缓存与验收](05_编码缓存与验收.md)：模型实际吃什么，怎样检查。

本文档中的数据路径均相对工作区 `/data/chenyiteng/projects/geowam-real`。

```mermaid
flowchart TD
 A[原始 RGB-D / sync / action] --> B[按 episode 筛选与时间对齐]
 B --> C[主相机 RGB: SAM 3.1]
 B --> D[相隔 4 帧 RGB: RAFT 双向光流]
 C --> E[本体与物体 mask / role]
 D --> F[光流 + 对应深度差]
 E --> F
 F --> G[有效性过滤 + UVD 编码]
 B --> H[双视角 RGB / 未来关节与夹爪]
 G --> I[冻结 Wan VAE / 标签下采样]
 H --> I
 I --> J[2405 个 .pt 窗口 + 五任务清单]
 J --> K[Video / Track / Action 联合训练]
```

训练缓存足够供现有模型加载；最终 mask／UVD 和 QA 包用于审阅、重编码及定位问题。原始数据包用于重跑 SAM／RAFT。
