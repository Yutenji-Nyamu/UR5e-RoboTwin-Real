# Metis4D 论文标题与文稿索引

## 标题

**英文：Metis4D: Learning Where and When to Focus for Task-Adaptive 4D World-Action Modeling**

**中文：Metis4D：面向任务自适应四维世界动作建模的时空聚焦学习**

Where 与 When 分别对应空间上交互显著的本体—物体局部对和时间上的耦合转变时刻；Task-Adaptive 指信息选择由任务指令与当前执行状态调制；4D World-Action Modeling 界定论文主体为四维世界动作模型。方法名 Metis 取自希腊神话中象征谋划与明智判断的名字，借指"判断何种信息对行动有用"，不展开为首字母缩写。

代码包、训练目录、checkpoint 与既有 Markdown 文件名保持不变；Track4D 仍为表示名称。

## 正文入口

- [Introduction](JanusAct4D-Introduction-中文.md)
- [Method](JanusAct4D-Method-中文.md)
- [Experiments 实验设计提纲](JanusAct4D-Experiments-实验设计提纲.md)
- [Overview 图注与说明](JanusAct4D-Overview-说明.md)
- [时空聚焦技术参考](Metis4D-时空聚焦技术参考.md)
- [研究记录](2026-09-20-Metis4D-面向交互后果的时空抽象-论文方案.md)

正文不包含方案讨论、工作记录、待填数字或未获得的实验结论；设计状态与代码核对信息集中在研究记录。

## 论文主线与贡献口径

论文回答一个中心问题：**策略应当从世界模型的预测中读取什么？** Track4D 三专家世界动作模型使这一问题有明确的读取对象；本体—物体耦合场及其导出的 CSIA 与 TAA 读取接口是对这一问题的回答，也是标题所指的核心贡献；异步视频条件是同一读取接口的扩展能力。

三项贡献按此主次组织：

1. **Track4D 中心的三专家世界动作模型（基础）。** 本体—物体三维位移场作为显式生成模态；异步噪声、干净动作条件与随机模态丢弃训练使同一模型支持联合生成、动作条件后果预测与多速率去噪。
2. **耦合场引导的世界读取（核心）。** 由 Track4D 定义本体—物体耦合场与耦合转变；Coupling-Salient Interaction Attention（CSIA）按交互显著度读取本体—物体局部对，Transition-Anchored Aggregation（TAA）将逐帧交互 token 聚合到锚定于耦合转变的少数时刻。在保留完整世界预测的前提下，动作专家对未来世界的读取被压缩为少量物理可检验的交互 token。其价值以等 token 预算的通用聚合器和运动幅度选择为对照判定。
3. **耦合转变接口的异步视频扩展（延伸）。** 保持目标机器人 Track 与 Action 的物理同步，为示范视频学习独立的进度映射，以已知时间变换监督阶段对应并约束节奏一致性。

Introduction 只在获得稳定结果后引用 Experiments 中最能支持核心主张的观察，不预写结论。

## 方法定稿前需冻结的实现项

以下设计尚未在代码中实现或验证，需依照正式实验配置实施并在验证集上冻结：

- Track4D 逐帧定义、采样间隔 $\Delta$、预测窗口内未来帧数 $N_f$ 与 VAE 时间压缩率 $\kappa$；
- 三独立噪声时钟的混合采样比例、丢弃概率、干净动作条件比例与各模态损失权重；
- 异步推理轨迹中 Action、Track、Video 的完成位置 $r_A<r_T\le r_V$；
- 子帧展开的解码损失权重 $\lambda_{\mathrm{unf}}$、耦合场邻域核 $\omega$ 与邻域大小；
- 读取接口的查询数 $K$、窗口范围 $[w_{\min},w_{\max}]$、显著度偏置强度 $\gamma_m$ 的初始化；
- 示范对齐模块的容差 $\eta$ 与损失权重。

代码版本的动作定义、Track 采样间隔和 VAE 时间粒度以正式实验配置为准；正文不使用与配置不一致的固定数值。
