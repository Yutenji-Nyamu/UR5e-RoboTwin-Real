# Metis4D：世界读取接口的技术谱系与选型

## 当前设计

Method §3 由三部分构成：由预测 Track4D 在 token 网格上计算的**本体—物体耦合场**（相对位移及其归一化耦合状态、耦合转变）；以耦合场导出的交互显著度为注意力偏置的 **Coupling-Salient Interaction Attention（CSIA）**；将逐帧交互 token 聚合到锚定于耦合转变的少数时刻的 **Transition-Anchored Aggregation（TAA）**。子帧时间展开为二者提供帧级时间轴。它是软聚合的信息瓶颈，不是硬 Top-k 稀疏注意力；完整世界预测保持不变，不据此宣称整个 WAM 获得稀疏计算加速。

核心理念是：动作应读取的世界信息由本体与物体之间运动关系的变化决定，而非由运动幅度、外观相似度或轨迹身份决定；该关系只有在本体与物体位移共处同一度量场（Track4D）时才可直接计算。以下按"理念来源"与"机制来源"分列，前者需在 Intro/Related Work 中正面比较，后者在 Method 中标注引用即可。

## 理念来源：以关系变化组织交互

| 论文 | 与本方案的关系 |
| --- | --- |
| Interaction Networks for Learning about Objects, Relations and Physics（NeurIPS 2016）；A Simple Neural Network Module for Relational Reasoning（NeurIPS 2017） | 以实体对的相对量作为关系表示的基本形式；本方案的耦合向量 $r$ 与 $[a_{\mathrm{object}};a_{\mathrm{body}};W_r\bar r]$ 属此类 |
| ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation（CoRL 2024） | 以关键点间关系约束刻画操作阶段；本方案的区别在于关系由生成的位移场自动计算，不依赖人工或 VLM 给出的约束 |
| Object-Centric Learning with Slot Attention（NeurIPS 2020） | 查询对空间区域的竞争性分配；CSIA 的查询—角色结构与此相近，但分配依据加入了物理显著度 |
| Gaussian Temporal Awareness Networks for Action Localization（CVPR 2019） | 可学习高斯时间核定位时段；TAA 的窗口形式与此一致，区别在于中心由耦合转变锚定而非自由学习 |
| The Information Bottleneck Method（1999）；Deep Variational Information Bottleneck（ICLR 2017） | "完整预测保留、决策读取压缩"分工的理论口径；reviewer 可能据此质疑收益来自压缩本身，需以等预算通用聚合器对照回应 |

## 机制来源：Method 各式对应的技术

| Method 中的构件 | 来源 |
| --- | --- |
| $K$ 个可学习查询聚合大量 token | Perceiver: General Perception with Iterative Attention（ICML 2021）；Flamingo（NeurIPS 2022，Perceiver Resampler）；BLIP-2（ICML 2023，Q-Former） |
| 查询由任务与状态调制 | FiLM: Visual Reasoning with a General Conditioning Layer（AAAI 2018） |
| $\gamma\log(s+\varepsilon)$ 形式的软偏置 | Masked-attention Mask Transformer for Universal Image Segmentation（CVPR 2022）的 masked attention（软化） |
| 内容得分与位置惩罚叠加于同一 softmax | Attention-Based Models for Speech Recognition（NeurIPS 2015） |
| 累积分布逆映射确定有序锚点 | 逆变换采样；与 Generating Sequences With Recurrent Neural Networks（2013）的单调位置注意力同属单调对齐 |
| 子帧展开 $\hat h_{n,i}=W_\tau h_{j,i}+p_\tau$ | Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network（CVPR 2016）的通道—空间重排，此处作用于时间维 |
| 子帧解码辅助损失 $\mathcal L_{\mathrm{unf}}$ | Understanding Intermediate Layers Using Linear Classifier Probes（2016）；Deeply-Supervised Nets（ACSIATS 2015） |
| 动作步时间对齐偏置（可选） | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation（ICLR 2022） |
| 三专家结构 | Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models（2024） |
| 异步噪声训练 | X-WAM |

## 最近邻（2025–2026）：必须正面比较的工作

核对日期 2026-09-21；arXiv 编号来自检索结果，引用前逐一核对原文与版本。

### 必读

| 工作 | 内容 | 与本文的关系 | 状态 |
| --- | --- | --- | --- |
| **Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning（Seeker，arXiv 2608.13422）** | 任务与状态条件查询在冻结 DINOv3 patch 特征上迭代细化，仅由动作损失学出 ROI；明确反对 PerAct 类关键帧／夹爪事件启发式 | 是"由动作损失学出的空间瓶颈"的现成形式，即等预算通用聚合器的最强版本；其"不预定义事件"的论点会被用于质疑 TAA。本文差异：显著度来自生成的三维位移场上的本体—物体耦合，连续可微、区分本体与物体，且作用于预测的未来而非当前观测 | 待读 |
| **EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon VLA Policies（arXiv 2606.20092）** | 在动作 chunk 隐状态上加关键帧预测头，输出未来 horizon 内逐步的关键帧概率，据此调度记忆写入；标签来自 Qwen3-VL 离线标注 | 与"在预测窗口内定位转变时刻"结构同构。本文差异：转变由生成的耦合场计算、无标注；用于聚合未来世界 token 而非写入历史记忆 | 待读 |

### 空间选择（与 CSIA 对照）

| 工作 | 内容 | 关系 |
| --- | --- | --- |
| Think Proprioceptively: State-Grounded Visual Token Selection for VLA Policies（2602.06575） | 指令分支＋本体状态分支双路 token gating，本体分支落在夹爪与接触区 | 仅本体侧线索，无物体侧运动与相对运动；可作"仅本体显著度"对照 |
| IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training（2609.00161） | 视频世界模型中把被操作物体的 cross-attention 转为 interaction map，重加权去噪监督 | 显著度来自语言—视频注意力而非几何；用途是训练加权而非策略读取；"interaction-aware world model"叙述需区分 |
| Training-Free Interaction-Aligned Visual Token Pruning（IAprune，2603.22991） | 语义响应与 2D 运动响应的空间一致性决定 token 预算与选择 | 运动幅度类基线的更强版本，训练无关 |
| Action-Aware Dynamic Pruning for Efficient VLA（ADP，ICLR 2026） | 按末端运动幅度切换剪枝：慢阶段保留全部 token | 运动幅度启发式的公开实现，作为探查 B 基线；其结论（运动小时更需细节）说明绝对运动幅度不是可靠依据 |
| Joint Hand Motion and Interaction Hotspots Prediction（OCT，CVPR 2022）；AFF-ttention（ECCV 2024） | 手／物体轨迹 → 交互热点分布 | "交互显著度"概念在视频领域的出处；热点由手轨迹条件预测而非相对位移计算 |

### 时间锚定（与 TAA 对照）

| 工作 | 内容 | 关系 |
| --- | --- | --- |
| Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining（KC-VLA，2603.01465） | 任务条件查询检测阶段边界，阈值触发 phase pointer，稀疏历史关键帧进 GR00T | 硬阈值、历史侧；TAA 的软锚点 $F^{-1}$ 与之对照 |
| KEMO: Event-Driven Keyframe Memory（2606.23589） | 事件驱动关键帧库，cross-attention＋负偏置初始化的门控残差注入 | 历史侧；门控初始化做法可用于 §4 示范融合 |
| SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models（2606.00664） | 只生成事件保持的关键帧（事件含夹爪开合符号变化），再插值 | 事件定义为夹爪状态；作用于生成效率。耦合转变包含且超出夹爪转变 |
| TTF-VLA: Temporal Token Fusion（AAAI 2026） | 像素差＋注意力相关度决定时间 token 融合，关键帧锚定防漂移 | 训练无关、历史侧 |
| CroSTAta: Cross-State Transition Attention Transformer（2510.00726） | 按学到的状态演化模式调制注意力权重 | 名称最接近 TAA，内容为历史状态转移建模 |
| Chain of World（CoWVLA，CVPR 2026） | 稀疏关键帧＋潜在运动链＋动作统一自回归 | 关键帧固定间隔，非事件锚定 |

### 世界→动作读取接口（贡献 2 的直接竞争者）

| 工作 | 内容 | 关系 |
| --- | --- | --- |
| LAWA: Latent Action as Intention Enables Efficient Future Imagination for WAMs（2608.24882） | video／latent action／action 三专家；结构化 mask 使动作专家只读潜在意图不读未来视频 | 架构最接近；接口为学出的 latent action，本文为物理定义的耦合交互 token。主表需其数字或同骨干复现 |
| LiLa-WAM（2608.03701） | 固定数量可学习查询压缩观测，联合读出未来特征与动作 | Q-Former 式压缩接口的 WAM 版本，等预算基线出处 |
| LaWAM（2606.15768） | latent action 解码为未来特征子目标条件策略；RoboTwin 91.2% | 同基准强基线，核对其 RoboTwin 协议 |
| Kairos: A Regret-Aware Native World-Action Model Stack（2606.16533） | 动作分支不依赖未来视频 token 的非对称注意力 | 非对称 mask 与本文 clean-action／读取接口需区分；"Kairos"名称已被占用 |
| KAM-WM（2607.04652） | Perceiver 压缩世界特征供策略读取 | 等预算基线 |
| TrajTok（CVPR 2026）；Trokens（ICCV 2025）；μ₀（2606.13769）；4D-WAM（2608.08023） | 见前表 | 表示与聚合思路的最近邻 |

## 备选阅读（未采用）

| 论文 | 说明 |
| --- | --- |
| Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention（2025） | 压缩、重要块选择、滑窗三路注意力。若 CSIA 的压缩 token 在精细接触阶段明显劣于全量读取，可让查询额外回读所选局部的原始特征；当前 Method 未包含此步，不引为来源 |
| DeepSeek-V3.2（2025，DSA） | 轻量索引器 Top-k KV 选择；解决长上下文成本，不建立本体—物体关系，不采用 |
| TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?（NeurIPS 2021） | 自适应空间权重形成少量 token 的早期形式；作为背景阅读 |

## 选型结论

主干采用耦合场引导的 CSIA+TAA。通用可学习查询聚合器（同子帧展开、同预算、无显著度偏置、自由学习中心）与运动幅度偏置是必须打赢的两个内部对照；若在等预算下无收益，先核查耦合场定义在真实 Track4D 上是否与接触事件对应（Experiments §3.2 预备测量），再调整显著度组合 $\phi$ 与偏置强度 $\gamma$。NSA 式局部回读仅在压缩 token 明显丢失精细接触信息时启用，并须明确引用。
