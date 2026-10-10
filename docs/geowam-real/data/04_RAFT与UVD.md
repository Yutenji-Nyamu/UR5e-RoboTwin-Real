# RAFT 光流与 Track-UVD

## 一句话理解

RAFT 找“某个像素下一时刻去了哪里”；深度补上“它离相机近了还是远了”。三者组成 UVD 位移。

## 计算顺序

1. 输入相隔 4 个原始帧、640×480 的主视角 RGB，相隔约 0.4 s。
2. RAFT-Large 同时计算前向与反向光流，更新 24 次。
3. 对起点 `(u,v)`，前向光流给终点 `(u+du,v+dv)`。
4. 在终点双线性读取下一帧深度，减起点深度得到 `dd`（米）。
5. 使用起终点深度有效、图像内、前后向误差 <1 px、本体／物体角色一致等条件筛出有效点。
6. 沿用作者 codec：du 和 dv 均按图像宽度归一，尺度 1/6、1/6、0.10 m；soft shrink 0.5 px／0.002 m；μ-law μ=31，编码为三通道图。

SAM 的实例编号随遮挡可能变化，因此最终过滤采用角色一致性；RAFT 与深度提供逐点对应，实例编号差异记录在诊断量中。

## 保存的结果

`geometry.npz` 包含编码图 `track_rgb`、连续编码位移 `track_delta`、有效区域 `valid`、本体／物体 `role`、前景和原始关键帧索引。空间大小 320×240。

`geometry.json` 记录覆盖率、静止深度残差、截断比例、实例编号变化等。`flow_review_*.jpg` 并排显示 RGB／RAFT／深度／UVD。

服务器还保留首／中／末的 `raw_pair_*.npz`，里面有全分辨率前后向 flow、深度差和前后向误差，用于进一步调试；本次云端几何包保存最终几何与预览，抽样 raw_pair 继续留服务器。

## 核验

已知 (+8,−4) px 平移：RAFT 中位误差 0.057 px，P90 0.124 px；已知 +0.02 m 深度变化、符号和编码检查通过。有效前景平均覆盖 79.94%，物体 86.76%。

本流程使用固定相机的像素位移与实测深度差、SDK 对齐和深度单位。实现见 [process_geometry.py](../../../scripts/geowam/process_geometry.py)、[qa_raft_uvd.py](../../../scripts/geowam/qa_raft_uvd.py)、[check_uvd_semantics.py](../../../scripts/geowam/check_uvd_semantics.py)。
