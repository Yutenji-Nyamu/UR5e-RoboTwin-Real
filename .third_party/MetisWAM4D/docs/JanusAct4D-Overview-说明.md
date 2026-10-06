# Metis4D Overview

## 图像与定位

当前版本：[overview-metis4d.png](../../JanusTrack4d_260824/overview-metis4d.png)。历史版本 [overview-v2.png](../../JanusTrack4d_260824/overview-v2.png) 与 [overview.png](../../JanusTrack4d_260824/overview.png) 保留。

标题与[文稿索引](JanusAct4D-论文标题与文稿索引.md)一致：

**Metis4D: Learning Where and When to Focus for Task-Adaptive 4D World-Action Modeling**

主面板 (a) 展示 Track4D 三专家世界动作模型；(b) 展开耦合场与 CSIA/TAA 读取接口；(c) 以较小的虚线面板展示异步视频条件扩展。图中的动作、轨迹和 RGB 位移场均为方法示意，不是实验结果。

## 中文图注

**图 1. Metis4D 总览。**（a）Track4D 在同一相机坐标系与时间轴下表示机器人本体与交互物体的三维运动；Video、Track 和 Action 三专家分别预测视觉变化、几何交互与机器人动作，并通过跨模态注意力交换信息。训练采用异步噪声、随机模态丢弃和干净动作条件，使动作与世界模态可以使用不同的去噪预算。虚线框展示 Track4D 的训练目标，不是推理时的输入。（b）由预测的 Track4D 计算本体—物体耦合场及其转变，导出交互显著度；CSIA 在显著度偏置下以任务条件查询读取本体—物体局部对，TAA 将逐帧交互 token 聚合到锚定于耦合转变的少数时刻；每个世界模态输出 $K$ 个交互 token 供动作专家读取，完整的世界预测保持不变。（c）在可选的视频条件设置下，同一接口通过状态条件的进度映射和已知时间变换监督，适配不同节奏的示范视频。图中的抓取、搬运与放置仅用于说明耦合转变的含义，不对应预设的阶段标签。

## English caption

**Figure 1. Overview of Metis4D.** (a) Track4D represents the 3D motion of the robot body and interacting objects in a shared camera frame and time axis. Video, Track, and Action experts predict visual evolution, geometric interaction, and robot control, exchanging information through cross-modal attention. Asynchronous noise, modality dropout, and clean-action conditioning allow actions and world modalities to use different denoising budgets. The dashed inset shows training targets of Track4D, not inference-time inputs. (b) A body–object coupling field and its transitions are computed from the predicted Track4D and yield an interaction saliency. Coupling-Salient Interaction Attention (CSIA) reads body–object pairs under a saliency bias with task-conditioned queries, and Transition-Anchored Aggregation (TAA) aggregates per-frame interaction tokens at moments anchored on predicted coupling transitions; each world modality provides $K$ compact tokens to the action expert while complete world prediction is retained. (c) In the optional video-conditioned setting, the same interface accommodates demonstrations with different timing through state-conditioned progress mapping and supervision from known time warps. Grasp, transport, and place illustrate coupling transitions rather than predefined stage labels.

## 读图约定

- 蓝色表示 Video／本体轨迹，青色表示 Track／物体轨迹，橙色表示 Action，紫色表示耦合场、CSIA/TAA 读取与时刻对应。
- Track4D 左侧的点轨迹帮助解释几何含义；实际生成表示为背景之外的 RGB 三维位移图，每个像素的 RGB 分别编码三轴位移。
- 主图箭头突出"世界特征 → 耦合场 / CSIA / TAA → 动作"的读取路径。clean-action 条件训练在训练条中表示，不增加绕过读取接口的世界到动作稠密路径。
- 每种世界模态产生 $K$ 个压缩 token；图中的时刻数与去噪条长度为结构示意，不是固定阶段数、固定步数或性能结论。当前图 (b) 仍按早期"成对聚合 + 有序时间中心"绘制，需重绘为"耦合场 → 显著度 → CSIA → TAA"的流程。
- 条件视频与目标执行使用独立进度轴，目标机器人 Track 与 Action 严格对应同一物理预测窗口。
- 当前交付为 PNG 位图；缩放和最终排版时需检查小字。正式投稿前应以矢量工具重绘。

## 生成记录

图像以早期 overview.png 为参考、使用图像生成工具编辑得到；以下为结构定稿时实际使用的提示，用于复核图中信息关系。

```text
Edit the provided figure. Preserve the exact title/subtitle, entire bottom panels (b) and (c), colors, robot imagery, denoising inset and training strip. Redraw ONLY the top panel (a) connections with a clean minimal graph. Current version has incorrect crossing arrows; REMOVE ALL EXISTING CONNECTOR LINES IN PANEL (a) before drawing the exact graph below. Do not carry any old orange or blue connectors over.

Top panel content has four zones, left to right:
ZONE 1: Current robot observation photo, label "Current conditions"; directly below photo list "RGB-D · masks · task · robot state". NO arrows from this photo to the Track4D illustration.
ZONE 2: Standalone dashed-outline card titled "Track4D representation". Keep blue robot body traces, teal object traces, three foreground RGB displacement maps labeled t1,t2,t3 and legend "RGB = dX, dY, dZ". Put "Training targets" in the bottom of this card. This is an INSET, not a forward-pass module. It has ABSOLUTELY NO arrows entering or leaving it. No line touching its border.
ZONE 3: A pair of small blocks "Video expert" (blue) and "Track expert" (teal, sublabel "Body + object"), vertically stacked. Put their forecast thumbnails in a compact strip below the pair, labeled "Future video" and "Track4D forecast"; DO NOT put the forecast thumbnails between the two expert blocks. Connect the Video block and Track block by one short vertical double-headed arrow labeled "MoT".
ZONE 4: A purple "Interaction focus" box, then orange "Action expert" to its right, then action-output robot thumbnails below Action.

The ONLY forward arrows within the model are:
- one BLUE arrow from VIDEO EXPERT directly to INTERACTION FOCUS;
- one TEAL arrow from TRACK EXPERT directly to INTERACTION FOCUS;
- one PURPLE arrow from INTERACTION FOCUS directly to ACTION EXPERT, labeled "Compact tokens";
- one orange downward arrow from ACTION EXPERT to its "Robot actions" thumbnails.
Absolutely NO orange feedback wires, no orange line above the experts, no arrow from Interaction focus back to any world expert.
Input-conditioning shorthand: add one small unobtrusive text line below the two world expert blocks or near the top-panel bottom, "All experts conditioned on current observation, task and robot state". No long condition rail or input arrows needed; the text states this shared conditioning.
Keep training strip exactly "Three-expert training" with "Asynchronous noise" | "Modality dropout" | "Clean-action conditioning". Clean-action conditioning is represented by this strip, NOT by any feedback arrows.
Keep "Different denoising budgets" inset exactly.
This simplified graph must have NO arrows touching the training-target inset, NO direct world-to-action bypass, and EXACTLY TWO arrows entering Interaction focus from world experts. Other details in panel (a) can be shifted slightly for whitespace. Academic figure, crisp readable high resolution. No new labels except those specified. Preserve bottom panels unchanged.
```

最后一轮局部文字修正：

```text
Make exactly one tiny typography correction to this image: in the dashed Track4D representation inset in panel (a), the header currently has a stray closing parenthesis. Replace the entire header with the exact text "Track4D representation" (no parenthesis or extra mark). Keep EVERYTHING ELSE pixel-faithful: all arrows, geometry, colors, positioning, panels, text, title, robot images and labels. Do not redesign, add, remove or reflow any other element. Return the corrected complete figure at the same resolution.
```

overview-metis4d.png 由 overview-v2.png 仅修改主标题与副标题得到，技术面板与 Track4D 表示名称未变。
