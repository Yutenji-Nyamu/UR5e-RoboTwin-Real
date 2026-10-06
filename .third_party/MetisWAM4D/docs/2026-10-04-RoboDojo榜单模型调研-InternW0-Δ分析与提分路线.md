# RoboDojo 榜单模型调研：InternW0-Δ 分析与提分路线（2026-10-04）

## 结论

1. 榜单上超过 OpenWAM-α（11.92）的模型中，**只有 InternW0-Δ（23.91）适合加载权重继续训练**：WAM 路线、视频专家同为 Wan2.2-TI2V-5B、RoboDojo 后训练权重与预训练 Base 均为 Apache-2.0。Awomo-0.5 权重可下但许可禁止训练；Simate-beta 权重 Apache-2.0 但是 VLA，结构不兼容；其余（PhysicalRSI、Rex-M1、VPP2、Liber-0）没有公开权重。
2. InternW0-Δ 相对我们融合设定（16.00）多出约 7.9 分，其中约 7 分来自 **Memory（34.0 vs 8.3）和 Open（10.2 vs 0）**；Generalization / Precision / Long-Horizon 基本持平。主因按证据强弱：**稀疏记忆上下文（锚帧 + 近期帧）**、**冻结 VLM 给 Action 提供场景语义**、**2 万小时预训练**；Causal Imprint、4D 蒸馏是次要项。
3. 对我们最值得借鉴的是：**(A) 稀疏记忆**（结构盲区，`cover_blocks` 单帧下无解）和 **(C) Causal Imprint 式的 Track 读取接口**（修复"Action 读生成 Track 有害"的矛盾，也是论文需要的证据）。
4. 推理侧已无提分空间（ckpt、权重均值、cfg、多采样、加轮、chunk 集成、执行步数均在噪声内）；不改结构时，唯一实测有效的手段是自采回灌。

## 一、榜单上超过 OpenWAM-α 的模型（2026-10-03 版）

来源：[RoboDojo 榜单](https://robodojo-benchmark.com/leaderboard) 前端数据（`assets/index-*.js` 内嵌的模型表与逐任务 JSON）、HF API、[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 上游 `policy/` 接入目录（本地副本 `RoboDojo/XPolicyLab` 尚未同步 `Awomo05`、`InternW0_delta`、`Simate_beta`）。

| 模型 | Score / SR | 路线 | 论文 | 代码 | 权重 | 可否继续训练 |
|---|---:|---|---|---|---|---|
| PhysicalRSI（港大 MMLab & KAI） | 36.27 / 31.38 | 智能体系统：VLM 规划，代码 / π0.5 / 记忆作工具，环境选择变体 | 项目页与报道 | 无 | 无 | 不适用 |
| Awomo-0.5 | 35.34 / 29.64 | 图像 WAM：FLUX.2-klein-4B + Qwen3-VL-4B（LoRA）MoT，20 槽历史 | 无 | XPolicyLab 推理代码 | `Auwomo/Awomo-0.5-Robodojo`（15.4 GB，可下载） | 否：LICENSE 为评测专用，禁止训练、微调、蒸馏 |
| Simate-beta | 33.95 / 27.96 | VLA：PaliGemma（SigLIP + Gemma）+ history encoder | 未发布 | XPolicyLab 推理代码 | `SIPAILab/sipai-robodojo-eval-ae`（13.5 GB，Apache-2.0） | 许可允许；无视频专家，结构不兼容 |
| Rex-M1 Preview（FT Lab & VisIncept） | 33.79 / 27.44 | 未知 | 未检索到 | 无 | 无 | 否 |
| VPP2-Preview（星动纪元） | 31.40 / 25.62 | 应为 WAM | 无 | 无 | 无 | 否（前作 VPP 开源，SVD 骨干） |
| Liber-0 Preview / Lite（LiberAI） | 30.74 / 25.52、29.24 / 24.23 | WAM | 仅融资新闻 | 无 | 无 | 否 |
| **InternW0-Δ（上海 AI Lab）** | **30.77 / 23.91** | **WAM：Wan2.2-TI2V-5B 视频专家 + ActionDiT + 冻结 RynnBrain-2B，30 层有向 MoT** | [arXiv 2609.31394](https://arxiv.org/html/2609.31394v1) | XPolicyLab 推理代码；训练代码"10 月底前发布" | `InternRobotics/InternW0-Delta-RoboDojo`（`robodojo.pt`）与 `InternW0-Delta-Base`（`pretrain.pt`），各 12.4 GB，Apache-2.0 | **是** |
| GPT-6-Astra | 28.97 / 22.48 | 闭源大模型直接控制 | — | — | — | 否 |
| DM0.5（Dexmal） | 24.90 / 19.34 | VLA | 技术博客 | `dexmal/opendm`（含训练） | `Dexmal/DM05`（Apache-2.0） | 许可允许；VLA |

ME-Brain-1.0（15.99）、GalaxeaVLA G0.5（14.88）、Xiaomi-Robotics-1（13.93）也在 Alpha 之上，均为 VLA，未逐个核实权重。

## 二、InternW0-Δ：设计与 RoboDojo 配置

论文与 `XPolicyLab/policy/InternW0_delta/config/eval_model.yaml`（随 `robodojo.pt` 发布的推理契约）：

- **结构**：Wan2.2-TI2V-5B 视频专家（30 层，hidden 3072）与 ActionDiT 动作专家（30 层，hidden 1024，随机初始化）逐层 joint attention；视频侧 T5 文本条件，动作侧 cross-attention 读冻结 RynnBrain1.1-2B 的多视角 + 指令隐状态与本体 token；推理只做一次视频 prefill（缓存 K/V），动作专家 10 步去噪，不生成未来视频。
- **稀疏记忆**（`memory.video: num_anchor_frames 1, num_recent_frames 1`）：本集第 0 帧 + 上一个动作块开始时（t−32 步）的帧 + 当前帧，三者的 Wan latent 都进视频专家。
- **Causal Imprint**（`future_delta`，loss 0.5；`semantic_alignment` 第 8 层对齐第 20 层未来特征，loss 0.1）：一组可学习 token 只看已观测帧，回归未来相邻 latent 差分并对齐未来特征；动作专家读这组 token，任何前向路径都看不到真实未来。
- **4D 蒸馏**：训练期 Track4World 教师 → 第 15 层视频 token 的 clip 级描述子 MSE，推理丢弃；RoboDojo 推理配置中未出现。
- **动作**：RoboDojo 原生 14 维绝对关节目标，映射进 80 维统一动作空间（其余维度屏蔽），z-score 归一化；32 步动作块，每执行 10 步重规划。
- **数据与训练**：预训练约 2 万小时（机器人 80% / Ego2Robot 10% / UMI 8% / Ego 2%），256×A800、24.5 万步、全局 batch 4096、约 14 天；RoboDojo 后训练只用官方 train，3 视角拼 384×256 画布，10 epoch，lr 5e-5。

## 三、差距落在哪里

| 维度 SR | InternW0-Δ（官方） | Ours 融合（本地 420） | OpenWAM-α（官方） |
|---|---:|---:|---:|
| Gen-Std / Gen-Random | 33.78 / 11.78 | 38.33 / 5.00 | 25.56 / 4.11 |
| Precision | 23.25 | 23.75 | 9.25 |
| Long-Horizon | 29.33 | 26.25 | 25.33 |
| Memory | **34.00** | **8.33** | 9.11 |
| Open | **10.17** | **0** | 1.08 |
| Overall | 23.91 | 16.00 | 11.92 |

逐任务（InternW0-Δ 取榜单逐任务 SR，我们取每任务 10 条的本地结果；括号为按维度权重折算到 Overall 的差）：

| 任务 | InternW0-Δ | Ours | 差 |
|---|---:|---:|---:|
| cover_blocks（Memory） | 92.7 | 0 | +3.1 |
| general_pickup（Open） | 62.7 | 0 | +1.6 |
| match_and_pick_from_conveyor（Memory） | 92.0 | 50 | +1.4 |
| make_kong（LH） | 47.3 | 10 | +0.9 |
| pour_liquid_into_cup std / random（Gen） | 68.0 / 52.0 | 40 / 0 | +0.7 |
| pour_balls_into_vase（Precision） | 54.0 | 30 | +0.6 |
| stack_blocks_by_language（Open） | 18.7 | 0 | +0.5 |
| stack_blocks（Gen） | 74.7 | 20 | +0.5 |
| fasten_screws（Precision） | 17.3 | 0 | +0.4 |
| build_tower（Precision） | 50.7 | 100 | −1.2 |
| classify_objects（LH） | 8.0 | 30 | −0.6 |
| make_toast std / random（Gen） | 24.0 / 8.0 | 60 / 0 | −0.2 |

官方全量评测与本地每任务 10 条只能近似对比；单任务差距在 10 条下有较大抽样误差，汇总到维度后结论稳定（Memory + Open 约占 7 分）。

## 四、性能来源判断

| 原因 | 证据 | 强度 |
|---|---|---|
| 稀疏记忆上下文 | RoboDojo 推理配置启用；Memory 34 vs 我们 8.3；LIBERO-Plus 消融 +3.9；榜单其他高分模型都带记忆（DM0.5 Memory 47.4 为 DM05-MEM，Awomo 20 槽历史，Simate history encoder）；我们只看当前帧，`cover_blocks` 三轮自采均 0 | 强 |
| 冻结 VLM 给 Action 场景语义 | 消融最大单项：+Qwen3.5-2B +6.9，换 RynnBrain 再 +8.7；Open 维度 `general_pickup` 63%、`stack_blocks_by_language` 19%，我们全 0（Action 只有 UMT5 文本，评测新指令无法落地到场景） | 强 |
| 2 万小时预训练 | 真机有无预训练 20% → 95%；RoboTwin Clean2Random 机器人预训练 4.4% → 32.3%；RoboDojo 后训练数据与我们 / Alpha 相同，Gen-Random 11.8 vs 我们 5.0 | 强（无法复现，可用权重） |
| 关节动作空间 | RoboTwin 无预训练对照：关节 41.3% vs EEF 32.1%；我们 EEF20 → cuRobo IK | 推测，RoboDojo 未验证 |
| Causal Imprint | LIBERO-Plus 合计 +7.4（对齐损失 +5.7） | 中 |
| 4D 蒸馏 | +1.9；描述子直接注入 Action 反而 77.8 < 78.4 | 弱 |

不是原因：每 10 步重规划（我们的执行步数对照 32 / 16 / 8 无差异，见 RoboDojo 评测记录"执行步数对照"）；输入分辨率（384×256 与我们 384×320 相近）。

**与我们发现的一致性**：4D 信息作为表征塑造的辅助监督有用、直接作为动作输入有害——与我们 hide_track 的结果一致（Action 读生成的 Track 退化；Track + Video 都对 Action 隐藏则 0/84）。问题不在"读未来"，而在读的是自生成的带噪未来；Causal Imprint 读的是确定性的、受未来监督的预测表征。

## 五、可借鉴项（预期收益为估计）

| 借鉴 | 预期 Overall | 成本 | 判断 |
|---|---:|---|---|
| **A. 稀疏记忆**：Video 上下文加锚帧（第 0 帧）与近期帧（t−32）latent | +2～4（Memory） | 中：Video 输入、时间 RoPE、attention mask，数据加载器取两帧；重做 RoboDojo 后训练 | 结构盲区，其他手段替代不了 |
| **B. 冻结 VLM 条件化 Action** | +1～2（Open + 语言定位） | 中：需离线 RynnBrain / Qwen-VL 权重，每次重规划多一次约 2B 前向 | 消融最大单项 |
| **C. Causal Imprint 式 Track 读取**：只看已观测帧的 token，用未来 Δuvd 与未来 Track 特征监督，Action 读这组 token | 不确定 | 中 | 对论文最有价值：让 Track 真正帮到 Action，解释并修复 hide_track 的矛盾 |
| D. 关节动作空间 | 不确定 | 中高：失去 Alpha 的 EEF 动作先验 | 先小规模对照 |
| E. 视频专家从 InternW0-Δ Base 初始化 | 可能大（鲁棒性） | 中：键名映射、Action 读取接口重新适配 | 比较对象从 Alpha 变为 InternW0-Δ |

不借鉴：自建 2 万小时预训练（算力差两个数量级）；GPT-6 执行期纠错（外挂，非策略本身；论文 Table 18 五个弱任务 10.8% → 47.2%）。

## 六、不改结构时的提分路线

已有证据：推理侧选项全部在噪声内；重点任务自采两轮配对 28 → 31 → 37 / 90，是唯一实测有效的手段。

| 路线 | 预期 Overall | 成本 |
|---|---:|---|
| v7：重点任务第 3 轮（round3 数据已采，`dataset_v4` 与配置已备） | +0～1.5 | 约 10 h |
| 自采扩到非重点但有信号的任务（`stack_bowls`、`make_toast`、`fill_pen_holder`、`classify_objects`、`sort_nesting_dolls`、`organize_table`、`hang_mugs` 等；`records_v1` 已有约 40 条未用的成功轨迹） | +1～2 | 每轮约 1 天 |
| 上两条合成单模型（35 任务示教 + 全部自采） | 同上 | 训练约 7 h + 420 条评测约 5 h |

最终报数前，把最佳模型与 Alpha 按每任务 20 条重评（标准误从约 1–1.5 分减半），比再挤 1 分更能支撑领先。论文的同数据对比目前是单模型 14.25 vs Alpha 9.58（+4.67）；融合版多出的部分来自自采数据，公平消融是让 Alpha 也用同批自采数据训练。

## 限制与待核实

- InternW0-Δ 视频专家的参数名能否直接映射到我们的 Video 专家（两边均为 DiffSynth 风格 `wan_video_dit.py`，可能另有 view conditioning / frame attention 模块），需下载 `pretrain.pt` 核对。
- InternW0-Δ 在本地 420 条协议下的成绩未测；官方 23.91 与本地协议只能近似比较。
- 关节动作对 RoboDojo 的收益、记忆与 VLM 在我们架构上的收益都未验证。
