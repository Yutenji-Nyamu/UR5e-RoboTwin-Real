# RoboDojo 后训练：数据方案与 RDJ_MetisWAM4D 数据集构建

> 日期：2026-09-28。对应 `plan.md`「训练课程 → S3 并行后训 → robodojo」。产物 `/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RDJ_MetisWAM4D/`，构建脚本 `scripts/data_prep/robodojo/rdj_track4d.py`。本文只覆盖数据；loader、损失改动与 `stage3_robodojo.yaml` 另记。

## 结论

| 决策点 | 结论 |
| --- | --- |
| 数据源 | 机器人几何（RGB ×3、head 深度、link id、内外参、关节状态）用 `GeoRobodojo_janustrack4d/<task>/train_4d/data/episodeN.hdf5`；末端位姿用官方原始 `RoboDojo_official_full/.../episode_N.hdf5`；文本缓存与指令用 `GeoRoboDojo_JanusAct4D_Imperfect` 的（同 RT2 的 `compact_prompt_sqlite_v1` 格式，35 条 prompt）。全部软链，不复制。 |
| 不用 Imperfect 的 track | 其 70/30 分桶是 JanusAct4D 论文的实验设计：qpos 桶只有机器人是 FK 真值，**两桶的物体轨迹都是 RAFT + 相机条件 DA3**（`object_method: native_da3_unfiltered_raft`），且 895 集（25.6%）物体掩码为空。v1 放弃物体轨迹，Imperfect 的 track / 深度 / 掩码整体不用。 |
| Track 表示 | 机器人像素的 t→t+4 像素系位移 (Δu px, Δv px, Δd m)，由 URDF FK 直接算（不调仿真、不调模型）；**非机器人像素一律"未知"**（role 0、编码为黑、不进 Track / role 损失），不能标静止背景——被操作的物体就在这些像素里。 |
| 离线 / 在线 | 用户选离线。单窗口在线成本实测 1.19 s（FK 0.45 + I/O 0.36 + 数学 0.18），本可在线；离线产物 lzf 压缩后每 269 帧 17 MB。 |
| 动作 | EEF20（每臂 xyz + rot6d + gripper），来自官方 `state/*_ee_poses`（**wxyz** 四元数，= link6 世界位姿，与 FK 对照位置差 0.1 mm、旋转偏置为单位阵）与 `state/*_ee_joint_states`；`action[t] == state[t+1]`（p99 差 1.6e-4）。注册表 `robodojo_arx_x5`（index 1，槽 0-9 / 34-43）。 |
| 相机 | head camera 一集内外参恒定 → camera token 单位阵标签，同 IG-10K robot。 |
| 划分 | 每任务编号最大的 3 集为 val：34 任务 × 3 = 102 集。 |

## 数据集

3,400 集 / 34 任务 / 1,814,728 帧（episode 176–1511 帧，25 Hz）。目录布局、`track4d.h5` 字段与窗口契约见数据集 `README.md`。关键字段：`delta_uvd [n-4, 240, 320, 3] f16`、`role [n, 240, 320] u8`（0 未知 / 1 机器人）、`eef20 [n, 20]`、`qpos [n, 14]`、`intrinsic_cv`、`extrinsic_cv`。

统计（`uvd_stats.json`，247 亿机器人像素；`eef20_stats.json` 为 train 划分 min / max）：

| 量 | p50 | p90 | p99 | p99.5 | p99.9 | 超出 codec 尺度比例 |
| --- | --- | --- | --- | --- | --- | --- |
| \|Δu\| px @320 | 0.25 | 17.8 | 68.8 | 84.5 | 109 | 1.9%（尺度 1/6 W = 53 px） |
| \|Δv\| px | 0.25 | 11.3 | 49.8 | 63.3 | 91 | 0.8% |
| \|Δd\| m | 0.0005 | 0.025 | 0.060 | 0.068 | 0.085 | 0.02%（尺度 0.10 m） |

RoboDojo 相机 f = 144 px（96° 视场）、机械臂距相机 0.27–0.44 m，同样米制运动在画面里比 RT2（f = 359 px）大得多，u 尾部更长；1.9% 裁剪与 RT2 本体的 3.0% 同量级，沿用冻结尺度，不为 RoboDojo 单独改 codec。

## 构建与核对

- 几何来源：janustrack4d 的 replay（`replay_timing = released_state`）已按发布关节状态渲染了机器人深度与 link id；本脚本只做 URDF FK（两臂基座 x = ∓0.3、y = −0.45、z = 0.765；夹爪 0..1 → −0.01..0.044 m 平移关节）与逐 link 刚体变换 + 重投影。
- 每集 4 帧对照发布的 stride-1 track（p99 ≤ 1e-4 m）：3,400 集全部通过，最大 2.3e-5 m。与 Imperfect qpos 桶 2,450 集的 stride-4 位移逐像素对照：0.03 px / 0.02 mm（f16 量化）。
- 对齐核对：`_qc/overlays/<task>_episodeN.png`（34 张，FK 轮廓 + Δu/Δv 箭头叠在真实 head RGB 上），轮廓贴合机械臂与夹爪。
- 算力：本机（load 40）16 进程 + d1 32 进程，共 42 分钟；每帧约 50 ms，主要是 Ceph 读 gzip 深度 / id 块。日志 `_logs/build_*.out`。

## 限制与已知问题

- **`stack_bowls` 100 集未构建**（`failed.json`）：其 replay 用了另一种时序（`low_level_joint_path_initial_then_10`），渲染帧与关节状态不对应（FK 差达 1.9 cm），且深度是全画面。要纳入需按 `JanusTrack4d_260824/preprocess/robodojo/repair_qpos_geometry.py` 的方式用 SAPIEN 重渲机器人-only 深度（d1 有 Vulkan），再跑本脚本。
- 物体与背景没有监督：RoboDojo 后训练里 Track 专家只收机器人像素（每帧约 13.7k / 76.8k = 18%）的梯度，Metis4D 的物体后果部分在此域不训不测。若要补，Imperfect `_pipeline/work` 里 2,605 集的 SAM 掩码 + RAFT 光流 + DA3 深度已算好，可作 v2 物体通道。
- `pour_balls_into_vase/78` 的 replay 文件缺 `joint_state` 组，脚本回退读官方关节状态（结果通过 FK 核对）。
- 官方 `action/*_ee_poses` 与下一帧 state 有 3 集出现 1.2–1.5 的单帧差（四元数符号翻转或跳变）；本数据只用 state，不受影响。`make_toast/72` 等 392 集 rot6d 单帧步长 > 0.15（最大 0.31，伴随 11 cm/帧的位置跳变），是原始数据本身的快动作或状态毛刺，未处理。
- 训练侧实现见下一节。

## 后训练：`stage3_robodojo`（2026-09-28 22:04 在 t2 启动）

配置 `configs/stage3_robodojo.yaml`，输出 `/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/stage3_robodojo/`，启动 `bash scripts/robodojo/launch_t2.sh`（tmux `rdj_train`，日志 `<host>_<时间>.log`，结束后由 tmux `rdj_keepalive_watch` 拉起保活）。

### 数据管线（`metiswam4d/data/robodojo/`）

- `RDJEpisodeDataset`：窗口契约与 RT2 完全一致（33 帧 / stride 4 / 32 步动作 / 三视角 L 布局 384×320），产出同名字段，直接复用 `RT2OnlineEncoder`、`collate_raw`、可视化器。
- **RGB**：`source.hdf5` 的 JPEG 是 BGR 字节序，PIL 解码后 `[..., ::-1]`；单测 `test_rgb_channel_order_table_is_warm` 用桌面红>蓝守住。
- **Track**：`delta_uvd`（px, px, m）→ `encode_uvd(..., tau_d = SHRINK_D_M["robodojo"] = 2 mm)`，与 RT2 同一颜色映射；有效 = `role > 0`；锚帧 = `role[s]`。未知像素（role 0）黑色、`track_delta` 0、不进前景。
- **动作**：Alpha RoboDojo 的 EEF20 是**逐臂基座系**（OpenWAM `robodojo_contract`：基座 `(∓0.3, −0.45, 0.765)`，基座四元数 wxyz `(0.707, 0, 0, 0.707)` = 绕 z 90°），我们的 `eef20` 是世界系。`eef_base.world_to_base_eef20`：`p_b = R_bᵀ(p_w − t_b)`，rot6d 两列左乘 `R_bᵀ`，夹爪不变；再用 Alpha RoboDojo `normalization_stats.npy` min-max 到 [-1, 1]，散布到槽 0-9 / 34-43。与 vendor `read_calibrated_eef20` 在 `stack_blocks/episode0` 前 64 帧逐元素对照 < 2e-4（`test_base_frame_matches_openwam_reader`）。
- **深度条件**：head 相机机器人像素渲染深度（mm），非机器人像素 0；`depth_stats.json` min 0.12 / max 0.70 m（150 集抽样，p0 0.124 / p100 0.682）。
- **未知区域不进任何 Track 损失**：`track_regions: {body: 1, object: 0, background: 0}`（role 0 落在 background 区，权重 0）；reader 的 role 交叉熵通过 `RT2EncoderConfig.unknown_role=true` 把无机器人像素的 token cell 目标写成 −1，`objectives.focus_aux_losses` 用 `ignore_index=-1`；unfold（子帧位移 L1）本来只算 role>0 的 cell。
- 文本：`episode_instructions_official.jsonl`（每任务 1 条）+ `text_cache/`（35 条 UMT5，RoboTwin 模板）。
- 验证集：`split=val` 的 102 集、固定窗口（`fixed_windows=true`），每 1000 步 4 batch/rank × 8 rank × 8 = 256 窗口，写 `val/*`；可视化面板也取 val 集。

### 初始化与课程

| 部件 | 来源 |
| --- | --- |
| Video / Action / proprio encoder | OpenWAM-Alpha-Sim-RoboDojo step 60000（825 / 824 / 2 张量） |
| Track、reader、focus attention、progress 三头、embodiment 嵌入、camera token | `rt2_direct_v2_uvd/checkpoints/step_0025003`（`initialize_from_exclude: [video., action., proprio_encoder.]`），同 uvd codec、接口已训 25k 步；只剩 RT2→RoboDojo 域迁移 |

选 RT2 uvd 的 Track 而不是旧 JanusTrack RoboDojo v3 step_29000：后者是相机系 xyz codec 且没有 reader / progress / camera 模块，接口要从零对齐（RT2 上这一段花了约 6k 步）。

- LR：video 3e-7（冻 500 + 升 1000）、action 3e-6（冻 500 + 升 1000）、track 3e-6（升 500）、focus 3e-5、proprio 1e-6（冻 500 + 升 1000）；constant。
- dense→compact 退火 3000 步、`max_bias 6`；损失 video 1 / track 0.2 / camera 0.1 / progress 0.05 / action 1 / unfold 0.1 / role 0.1 / condition 0.01；dropout 同 RT2。
- 8 卡 × bs 8 × accum 2 = 128 窗口/步，50k 步上限（≈ 3.8 个窗口 epoch），每小时 DCP ckpt + 可视化，保留 5 个。
- `num_workers` 12（116 核）：smoke 中 `time/data` = 0。

### smoke（`_smoke/stage3_robodojo_smoke`，12 步）

- 冻结期 4.65 s/步（enc 1.2 / fwd_bwd 3.1 / optim 0.3），显存 78–79.6 GB（nvidia-smi）；RT2 同配置全解冻后 9.6 s/步、`max_memory_allocated` 76 GB，预计 500 步解冻后回到该量级。
- 步 2–12：`action_shortcut_kept` 0.003–0.04（文本 + 本体在时 Alpha 动作先验直接可用，说明基座系 + 归一化正确），`action_shortcut_dropped` 0.1–0.5（需经世界接口读取，待训）；track_body 0.39–0.48（RT2 末期 0.19，域迁移），role 0.47→0.22。
- ckpt（`complete.json`）、val、可视化面板（帧带 RGB 正确、生成 track 落在机械臂）、代码快照均正常。

### 切到共钟方案：`stage3_robodojo_v2_coupled`（2026-09-29 16:41 在 t2 启动）

与 `rt2_direct_v3_coupled` 同一套改动（依据与实现见 `docs/2026-09-23-RT2直接训练-运行记录.md` 对应小节）：Track 与 Action 共钟 `{video 1.0, track 0.5, action 0.5}`、推理 20 轮第 10 轮出动作；`dense_read_floor 0`；`ReadActionHead` 探针（`read_action` 0.1）；LR video 5e-6 / track 1e-5 / action 2e-5 / focus 1e-4 / proprio 3e-6，从 13407 起 500 步升温。原 run 在 13600 停（最近 ckpt `step_0013407`，丢 193 步），新 run 从该 ckpt 完整恢复（含优化器），只有探针头 8 个张量新初始化。输出 `outputs/MetisWAM4D_260921/stage3_robodojo_v2_coupled/`，配置 `configs/stage3_robodojo_v2_coupled.yaml`。原 run 到 13k 步时：action 0.009、val/action 0.0113、track_body 0.245、read_ratio 0.018。

首条日志（13410）：action 0.066、track 0.217、video 0.070、read_action 0.32、3.2 s/步（升温期），GPU 96–100%。action 从 0.009 跳到 0.066 是 σ_A 分布随共钟改变的结果，与 RT2 一致，不是回退。

### 平台期与 `stage3_robodojo_v3_anneal`（2026-09-30 19:59 在 t4 启动）

v2 在 t2 训到 step 27530 停止（最新完整 ckpt `step_0027519`）。`loss/action` 20k → 27.5k 停在 0.0036–0.0038（2000 步分桶均值 0.00378 / 0.00377 / 0.00355 / 0.0036），`val/action` 仍缓慢下降（0.0011 → 0.0009，无过拟合），`focus/read_ratio` 0.44%。与 RT2 v3 同样是恒定 LR 的噪声平台。

续训 `configs/stage3_robodojo_v3_anneal.yaml` → `outputs/MetisWAM4D_260921/stage3_robodojo_v3_anneal/`，从 `step_0027519` 带优化器状态 resume，训 25k 步到 52500（≈ 33 h），配方与 `rt2_direct_v4_anneal` 相同（依据见 RT2 运行记录对应小节）：各组 LR 从 v2 的值在 27519 起余弦降到 1/10（`lr_schedule: cosine`、`lr_schedule_start: 27519`、`min_lr_ratio 0.1`）；`action_text` / `action_proprio` dropout 0.30 → 0.10；其余不变（含 `eval_every 1000` 的 val）。

启动：`NODE=t4 SESSION=rdj_train bash scripts/robodojo/launch_t2.sh configs/stage3_robodojo_v3_anneal.yaml -- --set init.resume_from=.../stage3_robodojo_v2_coupled/checkpoints/step_0027519`；t4 保活已由脚本停止，训练退出后按 `logs/t4_after_rdj_20260930_195924.pid` 拉回。日志 `stage3_robodojo_v3_anneal/a800bcctest0111-bd_20260930_195924.log`。

### 核对节奏

第一次日志（步 10）看损失量级与 `time/data`；步 500–1500 解冻期看显存与 `grad_norm`；第一次 ckpt（约 1 h）看 `vis/step_*/` 与 TensorBoard；之后每 1000 步看 `val/action`、`val/track_body`、`focus/read_ratio`。
