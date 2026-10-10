# SAM 3.1：从画面里找出机器人和物体

## 输入／模型

使用固定 head RGB，每隔 4 个原始帧取一个关键帧。模型为 `facebook/sam3.1` 的 multiplex 视频分割器；权重来源与路径见 [模型来源](../model/02_权重来源与初始化.md)。

每任务的提示分别描述 robot、方块、容器。物体提示结合实际颜色与形状，必要时首帧加框。模型从首帧提示沿视频跟踪。

## 输出

`masks_full.npz` 保存关键帧的 640×480 标签与角色：0 背景、1 本体、2 物体。这里的“物体”包含操作物和任务容器。实例标签保留用于诊断，模型的本体／物体监督采用角色。

`mask_review.jpg`：各提示的首／中／末帧叠图。`geometry.json` 记录每帧面积、实例数量、缺失初始目标和处理配置。

## 实际调整过什么

- block 容易把盒子也当成积木：改为具体颜色 cube，容器单独提示。
- 初帧只露夹爪时 robot 漏检：加几何框；关闭会绕过该框的批量 grounding 路径。
- 抽屉闭合后柜体漏分：对比 drawer／cabinet／wooden box，采用更稳定的 wooden box，并修复特定帧。
- 红方块部分遮挡及盒内漂移：重新提示，结合当前帧颜色区域辅助；保留修复配置。

最终针对 12 条记录、44 个提示帧修复，14 条纸盒数据另作专项重处理。72 条首中末掩码已检查。

## 配置／脚本

[perception.json](../../../configs/geowam/perception.json) 保存任务提示、框、逐条修复与阈值；[process_geometry.py](../../../scripts/geowam/process_geometry.py) 运行分割；[repair_masks.py](../../../scripts/geowam/repair_masks.py) 应用修复。提示变化后根据内容签名重建几何与缓存。
