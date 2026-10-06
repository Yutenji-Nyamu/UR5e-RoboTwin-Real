# RoboDojo Memory / Open 提分：InternW0 启发方案（2026-10-04 起）

代码目录 `metiswam4d_inspired_by_internw0/`（不改 `metiswam4d/`）。前置调研见 `2026-10-04-RoboDojo榜单模型调研-InternW0-Δ分析与提分路线.md`。

## 目标与分工

- 现状：融合 16.00 / 21.32（33 任务 anneal 34962 + 9 重点任务 v6 tail3，本地 420 条协议、逐步观测）；InternW0-Δ 官方 23.91。
- 分治：Memory（6 任务）+ Open（8 任务）共 14 个任务由新模型负责、单独训练与评测；其余 28 个任务沿用现有融合结果，最终 420 条两部分合并统计。
- InternW0-Δ 的优势集中在 4 个任务（其余由维度均值反推）：

| 维度 | 拉开差距的任务（InternW0-Δ / 我们） | InternW0-Δ 在同维度其余任务 |
|---|---|---|
| Memory 34.0 vs 8.3 | `cover_blocks` 92.7 / 0、`match_and_pick_from_conveyor` 92.0 / 50 | 4 个合计约 19 |
| Open 10.2 vs 0 | `general_pickup` 62.7 / 0、`stack_blocks_by_language` 18.7 / 0 | 6 个约 0 |

  这 4 个任务追平 InternW0-Δ 时，融合约 22–23。

## 决策（2026-10-04 用户拍板）

| 项 | 决定 |
|---|---|
| 结构 | 仍为 Video / Track uvd / Action 三专家；不引入 VLM |
| 初始化 | Video、Action、proprio ← InternW0-Δ `robodojo.pt`；Track ← RoboDojo v6；Action 对 Track 永久隐藏（v6 结论） |
| 记忆 | Video 专家输入本集第 0 帧 + 最近 4 个 chunk 窗口的首帧（干净上下文 latent），训练时随机少给，推理时不足也可用 |
| 动作 | 采用 InternW0 的 14 维绝对关节、z-score（其 `dataset_stats.json`），80 维槽 `[0:6] [16] [40:46] [56]`；评测走官方 joint 控制 |
| 画面 | 采用 InternW0 的 384×256 单画布：头相机上 2/3（384×170），左右腕部下 1/3 各 192×86；RGB 直接取 Isaac 输出 |
| 数据 | Memory 6 任务的官方示教全部使用；4 个缺口任务额外生成成功轨迹（视频、掩码、机械臂 / 夹爪 / 物体四相位 uvd） |
| Open 数据 | 允许为 Open 任务生成训练数据；上榜或论文需注明 Open 不再是零样本 |
| 教师 | 先用 InternW0-Δ 原版（独立 Python 3.11 环境）在协议外布局滚动；环境不可用时退到特权脚本专家 |
| 机器 | t2（8×A800）；采集与仿真阶段保活常开，只在正式训练期间停保活 |

## InternW0-Δ 推理契约（XPolicyLab `5b06740`，`policy/InternW0_delta/`）

- 本地代码：`/m2v_intern_v3/danglingwei/files/InternW0_delta_src/XPolicyLab/policy/InternW0_delta/`（仅推理，训练代码 10 月底发布）。
- 权重：`robodojo.pt`（step 9090，sha256 `344d42…8aaa`）、`pretrain.pt`（sha256 `a6bcf7…a08`）、RynnBrain1.1-2B，存放 `/ytech_milm_intern/danglingwei/model_zoos/InternW0-Delta/`。
- 结构：Wan2.2-TI2V-5B 视频专家（30 层 3072）+ ActionDiT（30 层 1024），24 头 × 128，与 OpenWAM-α 同尺寸；视频注意力 `first_frame_causal`；Causal Imprint（`future_delta` 查询 token + 语义对齐）。
- **动作专家的 cross-attention 已整体替换为读 RynnBrain 原始隐状态**（`configure_vlm_conditioning`：K/V 按 VLM 宽度新建，Q/O 从文本通路复制）。不接 VLM 时这些 K/V 需改回读 UMT5 文本并重新训练。
- 推理：动作块 32 步，每执行 10 步重规划，10 步去噪；每次 `get_obs` 后 `update_obs`；状态 `joint_action.vector` 14 维；指令取 `task_instruction` / `instruction`。
- 画布：三相机先各自 bilinear(antialias) 缩到 240×320，再缩到画布区域拼接，`x*2-1`。
- 记忆：保留第 0 帧与最近 33 帧；锚帧 = 第 0 帧，近期帧 = 当前重规划前 32 步的帧。

## 环境与下载

- 网络（2026-10-04 实测，t2）：海外代理 `oversea-squid2.ko.txyun:11080` 可达 huggingface.co / PyPI / download.pytorch.org / GitHub（各 <1 s）；huggingface.co 单流约 1 MB/s、48 路约 25–30 MB/s。国内代理 `10.68.41.218:11080` 到 PyPI / GitHub 很慢，HuggingFace 不通。
- 下载：`metiswam4d_inspired_by_internw0/scripts/download_internw0.sh`（分段并发、可续传、sha256 校验），tmux `iw0_download`，日志 `model_zoos/InternW0-Delta/_download.log`。
- InternW0 运行环境：`/ytech_milm_intern/danglingwei/envs/internw0-delta`（uv venv；torch 2.10.0+cu128、transformers 5.13.0、flash-linear-attention 0.5.1），脚本 `scripts/build_internw0_env.sh`，tmux `iw0_env`。共享 `/usr/bin/python3.10` 不动。
  - 整个环境在共享云盘上：venv、基础解释器（`envs/uv_python/cpython-3.11.15-linux-x86_64-gnu`）、uv 缓存（`envs/.uv_cache`）都在 `/ytech_milm_intern/danglingwei/envs/`，训练机重启或换同镜像的机器后直接可用，不需要重装。
  - 依赖机器镜像的只有：`gcc` 与 `/usr/local/cuda-12.8`（RynnBrain 的 FLA/Triton 内核首次前向时编译），以及仅在重建环境时用到的 `/usr/local/bin/uv`。新机器先核对这三项；缺失时重跑构建脚本（幂等，已下载的包走云盘缓存）。

## 计划

| 步骤 | 内容 | 状态 |
|---|---|---|
| S0 | 下载权重、建 InternW0 环境、教师服务冒烟（1 个缺口任务 1–2 条，含 4D 录制） | 进行中 |
| S1 | ~~InternW0-Δ 原版本地对标~~：用户取消（耗时过长）；对标沿用官方榜单数字，教师成功率由 S2 采集结果给出 | 取消 |
| S2 | 教师在 seed 1/2 布局滚动 4 个缺口任务（不碰 seed 0 布局 0–9），录真值深度 / 分割 / 物体位姿；转换为新数据集；Memory 6 任务官方示教转换 | |
| S3 | 新目录实现：InternW0 键名映射、记忆帧、关节动作、新画布、三角色 Track、单测（含 RGB 顺序） | |
| S4 | t2 训练 | |
| S5 | 新模型评 14 任务 140 条，与 28 任务现有结果合并成 420 条 | |

## 实现（`metiswam4d_inspired_by_internw0/`，只 import `metiswam4d`，不改原代码）

| 模块 | 内容 |
|---|---|
| `joint_action.py` | InternW0 的 14 维关节 z-score（`dataset_stats.json` global mean/std，裁剪 ±5）↔ 80 维槽 `[0:6] [16] [40:46] [56]`；关节 → `take_action` 字典 |
| `data.py` | `IW0EpisodeDataset`：画布 384×256（与 InternW0 `build_training_exact_canvas` 逐像素差 ≤1）、记忆帧 `[0, s-128, s-96, s-64, s-32]`（钳到 ≥0，训练时每个历史槽 15% 换成第 0 帧）、动作 = 官方 `joint_action/vector[s:s+32]`（= 下一帧关节状态）、本体 = `joint_state[s]`；Track 沿用头相机 uvd（role 3 夹爪并入本体）。`IW0OnlineEncoder`：记忆帧逐帧单独编码后接在窗口 latent 前，`video_clean` = 5 记忆 + 3 窗口 = 8 个 latent 帧 |
| `model.py` | Video ← InternW0 `mixtures.video.*`（825 张量，不含 Causal Imprint 的 `future_delta_*`）；Action ← `mixtures.action.*`（`head`→`action_decoder`），cross-attn 的 K/V/norm_k 与 text_embedding 取 Alpha RoboDojo 的 UMT5 通路，共 824；proprio ← InternW0 `proprio_encoder`（type_embedding 并入 bias）；Track / reader / progress / embodiment ← v6 tail3（689）。Video 干净前缀 6 帧；reader 前置钩子去掉记忆帧；`ActionSampler`：Action 只读干净世界（记忆 + 当前帧 + Track 条件），10 步 |
| `train.py` | 复用 `metiswam4d.train.train`，替换 dataset / encoder / build / init / 加噪与 Video 损失前缀 / 动作分组诊断 / 可视化，代码快照包含本目录 |
| `visualize.py` | 当前窗口帧带与 Track、动作曲线（部署路径 `vis/action_mse_deploy`）、记忆帧带 |
| `eval/` | `iw0_server`（InternW0 原版，每个仿真 episode 一个 session，一卡多客户端）、`mem_server`（本模型）、`iw0_deploy` / `iw0_client`（官方逐步观测节奏；可选 4D 录制：头相机 GT 深度、instance id、场景实例位姿）、`iw0_campaign`（复用 `rdj_campaign` 的轮次 / claim / 看门狗 / 汇总） |
| `data_prep/` | `build_gt4d.py`（录制 → episode：SAPIEN 机器人几何 + FK 本体 Track + GT 刚体物体 Track，role 0 静态 / 1 臂 / 2 物体 / 3 夹爪）、`assemble_dataset.py`（官方 Memory 示教 + 教师 episode → index / instructions / UMT5 缓存）、`qc_gt4d.py` |

### 已验证

- 数据单测 5 项：记忆索引、关节往返与槽位、画布与 InternW0 运行时一致、RGB 顺序（桌面红 > 蓝）、`joint_action[t] == joint_state[t+1]`。
- 全尺寸模型 CPU 装载：Video 825 / Action 824 / v6 689 张量，0 缺失 0 形状不符，抽查张量与 `robodojo.pt` 逐位相同。参数：Video 5.0B、Track 0.69B、Action 1.02B。
- 小模型训练冒烟（t2 单卡，`configs/iw0_smoke_tiny.yaml`）：数据 → 编码 → 损失 → val → ckpt → 可视化全链路跑通，`time/data` 0。
- InternW0 服务离线探针：首次 `act` 82 s（Triton/FLA 内核编译，每进程一次），之后 0.89 s；在 `cover_blocks` 首帧上预测关节与录制的下一帧状态平均差 0.001 rad。
- `dataset_v1` 官方部分：6 个 Memory 任务 × 100 集（97 train / 3 val），`cover_blocks`、`match_and_pick_from_conveyor` 用已有物体 Track 叠加（RAFT + DA3），其余为机器人 FK。官方示教每个任务只有一条固定指令。

## S2 教师采集（2026-10-04 23:07 起，t2）

- 冒烟（`iw0_teacher/smoke_cover`，seed 1，`cover_blocks` 2 条）：2/2 成功；4D 录制→转换→QC 通过：物体 = 本集实际移动 > 2 cm 的实例（cover_blocks 为 3 个杯子，方块不动），GT 刚体位姿质心重投影误差中位数 1.1 px（位姿与相机外参坐标系一致）；QC 叠图 `datas/IW0_MemOpen4D/_smoke_gt4d/qc_cover_blocks_ep0.png`（臂蓝 / 夹爪青 / 物体红，Δuv 箭头）。
- 采集速度：逐步观测 + 4D 录制约 2 步/s（`take_action` 0.26 s、`get_obs` 0.14 s、策略摊销 0.1 s、录制 4 ms）。InternW0 服务每进程首次推理需编译 Triton/FLA 内核（82 s，之后同机复用缓存约 6 s），冷启动加载约 10 分钟（Ceph 上读 28 GB 权重）。
- 正式采集：4 个缺口任务 × {seed 1 布局 0–54、seed 2 布局 0–54、seed 0 布局 10–54} = 620 条（`collect_t1_seed{1,2,0}`，GPU 0–2 / 3–5 / 6–7，每卡 1 个 InternW0 服务 + 2 个 Isaac 客户端）。慢任务（cover / conveyor / stack_by_language）另开后半段布局（seed 1/2 的 28–54、seed 0 的 32–54；`collect_t1_seed{1,2,0}b`），客户端共享同卡已有服务（`--attach-base-port`，服务按 episode 分 session），共 21 个客户端，每卡显存 ≤ 54 GB、利用率 81–100%，保活常开。后半段 campaign 首次启动时客户端端口用错（连到不存在的端口，约 40 分钟无产出），已修正并重启。
- 转换：`convert_loop.sh`（tmux `iw0_convert`，每 20 分钟一轮，episode 编号 = seed 偏移 + 布局号，前后半段重复的布局按同编号去重）。
- 教师成功率低于官方榜单（截至 00:10：general_pickup 13/58、stack_blocks_by_language 6/40、cover_blocks 18/24、conveyor 25/28；榜单 62.7 / 18.7 / 92.7 / 92.0），原因未确认；只影响成功轨迹数量。

### 教师成功率低于榜单：排查（2026-10-05 00:20–01:30）

| 假设 | 检验 | 结论 |
|---|---|---|
| 采集布局（seed 1/2）比评测布局难 | 按 seed 拆分：general_pickup seed1 5/21、seed2 5/25、seed0（布局 ≥10）7/23；协议布局 0–9 定向测试（不录 4D）2/10 | 排除 |
| 4D 录制的深度 / 分割 annotator 影响渲染 | 上面的协议布局测试不开录制，同样 2/10 | 排除 |
| 本地 RoboDojo 版本（9/8 gitee 快照）落后 | 上游 9/8 之后相关改动只有两处：9/12 取观测前渲染同步（与我们 kit 参数 + `get_obs` 前 `render()` 等效）、9/16 修 stack_blocks_by_language 指令缺空格（"order ofblue"）；环境 / 控制 / 任务配置无其他改动 | 排除（指令笔误只影响该任务的语言） |
| 画面滞后于物理状态 | 教师录制：画面 t 与状态 t 的机器人掩码吻合度 0.79，与 t±1 约 0.50 | 评测环境给的是当前帧，排除 |
| InternW0 记忆帧取错 | 逐行核对 `_build_memory_inputs`：第 t 步重规划取第 t−32 步帧和第 0 帧；我们发送的帧（每 10 步余 8、重规划步、第 0 步）正好覆盖 | 排除 |
| RynnBrain 内核数值错误 | causal-conv1d 快速路径 vs 回退：last_hidden_state 相对差 2.5%，逐 token 余弦 ≥ 0.999，重复计算完全一致 | 排除 |
| 我们的服务封装 / 部署与官方路径不等价 | 官方 `setup_policy_server.py` + 官方 `deploy.py` 对照：已接通，但与采集同卡显存超限，为保护采集暂停 | 采集结束后补测 |

失败形态（general_pickup 54 条失败全部为 200 步超时、得分 0）：机械臂到达目标附近反复调整抓取，200 步内提不起来；成功的 episode 63–199 步，多条接近上限。

### 顺带发现：官方旧版数据画面滞后一帧（影响所有基于 RDJ 的模型）

官方 2026-09-15 重新发布数据（HF `RoboDojo-Benchmark/RoboDojo` PR #39/#42–#46）："Color drops frame 0 and shifts forward"，即旧版第 t 帧画面对应第 t−1 步状态。RDJ 用的是旧版（BGR 字节序 JPEG）。实测 RDJ：画面 t 与状态 t−1 的机器人掩码吻合度 0.877，与状态 t 只有 0.529。评测时画面是当前帧，所以此前所有 RDJ 训练的模型（含 16.00 融合版）训练与部署之间都差一帧。新模型数据集已修正：官方 episode 的视频帧、记忆帧、RGB 条件取 t+1（`OFFICIAL_RGB_SHIFT`），教师 episode 不偏移；单测 `test_official_rgb_shift_aligns_image_with_state`。

## S2 收尾与 S4 启动（2026-10-05）

### 10-04 夜间采集结果

t2（`a800bcctest0125-bd`）10-05 08:06 被回收前，6 个采集 campaign 基本完成（seed1 219/220、seed2 220/220、seed0 179/180，后半段全部完成）。转换后成功 episode：cover_blocks 120、match_and_pick_from_conveyor 143、general_pickup 41、stack_blocks_by_language 18（共 322）。Memory 两任务教师成功率 76–96%，general_pickup 约 26%，stack_blocks_by_language 4–18%。官方部署路径对照 `check_pickup_official` 0/10 完成，未得出结论。

### 物体 Track：其余 4 个 Memory 任务 + 帧对齐

- 官方示教的物体 Track 不重跑模型：Imperfect `_pipeline/work` 已有全部 6 个 Memory 任务的四相位 RAFT、DA3 和 Qwen3.5-9B + SAM3.1 掩码，`scripts/data_prep/robodojo/rdj_object_track.py` 在 CPU 上合成。
- `--rgb-shift 1`：物体部分取画面 t+1→t+5（= 状态 t→t+4），与 FK 机器人 Track 和 `OFFICIAL_RGB_SHIFT` 后的画面对齐；v2 overlay 的物体部分比机器人晚一帧。
- 物体掩码为空的 episode（swap_T 98/100、imitate_sorting_sequence 23/100，SAM 定位失败）逐集回退为光流阈值判定，机器人 FK 掩码先膨胀 3 px 去掉臂缘噪声（cover_blocks 原本就是光流判定，同样加膨胀）。
- 输出 `RDJ_selfplay/object_track_v3_shift1/`，600 集 0 错误。物体像素 / 转移中位数：conveyor 1409、cover 143、swap_T 135、press 58、sorting 52、swap_blocks 20。叠图 `object_track_v3_shift1/_qc/object_overlay_qc.png`：被搬运的 T 块、方块标注正确；臂靠近相机时 FK 掩码外的边缘、sorting 示范区有成片标注，精度与 v2 同级，只作辅助监督。
- `assemble_dataset.py`：`train_4d` 软链指向新根；`--teacher-val`（每任务编号最大的 3 条教师 episode 留作验证，Open 任务此前没有验证集）；`--task-repeat`（Open 教师 episode 重复 4 次）。10:28 试组装：训练 1202 行 / 验证 30 行，8 个任务各取一个窗口读取通过。

### Open 补采：重跑失败布局

- 官方布局只有 seed 0/1/2 各 55 个，已全部用过；教师策略无种子，失败布局的场景不在数据集里，重跑即得新场景的成功轨迹。排除已成功布局后剩 251 集（general_pickup 124、stack_blocks_by_language 127）。
- `rdj_campaign` 同一 task config 同时只给一个客户端，每个 campaign 只有 2 个客户端在跑；按布局交错切成 11 个块 campaign（`iw0_teacher/collect_open_seed{1,2}r_c{0..3}`、`collect_open_seed0r_c{0..2}`，每卡 1 个服务 + 2 个客户端）：t3 8 卡、开发机 2 卡、d1 1 卡，10:20 启动。录制 `records_t1/seed{1,2,0}r`，编号偏移与原 seed 相同（`convert_loop.sh` 已加入）。

- 11:30 中途：重跑失败布局的成功率很低（general_pickup 5/111，stack_blocks_by_language 3/76，约 4%；首轮约 26% / 4–18%），这些布局本身对教师更难。用户决定跑完再开训。
- d1（IDC，驱动 535.54）上 Isaac 客户端启动后 1 h 无 episode（看门狗重启后同样卡住），`collect_open_seed0r_c2` 11:33 停止，其 23 个布局不采。Isaac Sim 5.1 此前只在 BCC 机器（驱动 535.161）上跑通，判断为驱动版本问题（未核实）；RoboDojo 闭环评测不放 d1 / t1。

- 其余 10 个块 11:58 前结束；`collect_open_seed1r_c3` 的 general_pickup 最后一个布局客户端每次退出码 0 但不产出 episode，manager 无限重试，交接链一直等它，t3 7 卡空转（仅保活）到 15:30 才被发现并停掉。补采合计新增 general_pickup 5、stack_blocks_by_language 5。
- 15:33 组装：Open 教师 general_pickup 43 训练 + 3 验证、stack_blocks_by_language 20 + 3（各 ×4）；15:35 t3 tmux `iw0_train` 启动 8 卡训练（保活按 pid 停止，gpu_filler 开），日志 `outputs/MetisWAM4D_260921/iw0_memopen_v1/a800bcctest0125-bd_20261005_073524.log`。装载 InternW0 video 825 / action 824 张量，0 缺失。

### InternW0-Δ 官方部署对齐（2026-10-05 t2）

- 目的：本地教师 general_pickup 约 26%（协议布局 2/10），榜单 62.7%；分清是我们的服务封装问题还是本地协议更严。做法：官方 `XPolicyLab/setup_policy_server.py` + `deploy.yml`（PYTHONPATH 需含 `InternW0_delta_src` 本身，否则 `No module named 'XPolicyLab'`）+ 客户端 `METIS_IW0_OFFICIAL_DEPLOY=1`（官方 `deploy.py`），与我们的封装在同一批协议布局（seed 0 的 0–9）上各跑 general_pickup / cover_blocks 10 集，目录 `iw0_teacher/official_t2/`。
- **Isaac 不能跑在 CUDA MPS 下**：t2 上 VLABench 评测留下的 MPS 守护进程使 Isaac 客户端约 9 s/步（无 MPS 时约 0.5 s/步）。16:35 停检查、按 pid 停保活、关 MPS（`quit` 卡住时 TERM 掉 mps-server）、以原参数重启非 MPS 保活后重跑。RoboDojo 的采集 / 评测机器上不开 MPS；VLABench 评测（MuJoCo 在 CPU）不受影响。
- 新机器首次启动 Isaac 编译着色器约 14 min，之后走缓存。
- 结果（协议布局 seed 0 的 0–9，各 10 集）：

| 任务 | 官方部署 | 我们的封装 | 榜单 |
|---|---|---|---|
| general_pickup | 3/10 | 1/10（另一轮 2/10，合计 3/20） | 62.7 |
| cover_blocks | 4/10（score 50.5） | 4/10（score 50.5） | 92.7 |

  本地协议下 InternW0-Δ 原版远低于其榜单分，cover_blocks 官方与我们的封装完全一致，差距不来自封装；general_pickup 官方部署略高（样本小）。比较对象改为「本地协议下官方部署的 InternW0-Δ」，榜单数字只作参考；之后的教师采集用官方部署。

### 本地评测检修（2026-10-05 18:00 起，t2 GPU 4–7）

- 我们的 Isaac 客户端外壳（`rdj_client` + `rdj_campaign.client_command`）相对官方 `src/eval_client/main.py` 只多：观测带相机内外参、视频集数上限、cuRobo 捕获 CUDA graph 前的 OptiX 预热、三个渲染同步 Kit 参数（`checkForHydraRenderComplete=1000`、`renderer/waitIdle`、`hydraEngine/waitIdle`）。步数上限来自官方任务代码（general_pickup 200、cover_blocks 800）；官方 `_task.yml` 的 `eval_nums` 为 50（我们的协议每任务 10 集）。InternW0 README 自测 54 任务 6300 集总 SR 22.92%，与榜单总分 23.91 一致。
- 纯官方客户端（`eval_policy.sh` 默认 10 个并行环境，及单环境 `main.py`）在本机都停在 `Resetting all environments`（不做 OptiX 预热时 cuRobo 图捕获挂起），无法直接对照；外壳里的预热是必要的、不影响画面与策略。官方 InternW0 包按其 README 链入 `RoboDojo/XPolicyLab/policy/InternW0_delta`（软链）。
- 画面：教师录制（外壳）与官方旧版示教同任务首帧平均亮度系统性不同：cover_blocks 我们亮约 15%（124/94/80 vs 108/81/69，20 集标准差 < 1），conveyor 我们暗约 4%。对照图 `iw0_teacher/official_t2/frame_compare_cover.png`。
- 假设：RDJ 旧版数据画面比状态晚一帧，上游 9/12 才加取观测前渲染同步；若 InternW0-Δ 在旧数据上训练、在未同步的评测上出榜，我们强制同步反而与其训练分布错一帧。对照：同一 50 个协议布局（seed 0 的 0–49，与官方 eval_nums 一致），官方 deploy，外壳带同步（`wrap50_*`）vs 去掉同步（`nosync50_*`，`METIS_KIT_RENDER_SYNC=0`）。18:55 时 general_pickup 带同步 6/14。
- 结果（20:08）：

| 任务（seed 0 布局 0–49，官方 deploy） | 带渲染同步 | 不同步（官方客户端默认） | 榜单 |
|---|---|---|---|
| general_pickup | 13/50（26%） | 19/48（40%） | 62.7 |
| cover_blocks（进行中） | 14/19，score 81.6 | 15/17，score 90.3 | 92.7 |

  渲染同步明显压低 InternW0-Δ；不同步 + 50 集下 cover_blocks 接近榜单。另一项是样本量：协议 10 集与 50 集差别很大（general_pickup 带同步在布局 0–9 为 3/10、0–13 为 6/14）。本地评测改为不同步、每任务 50 集（官方 `eval_nums`）后再与榜单对比。
- **结论：本地评测与当前官方协议一致，不是 bug。** 上游官方仓库（GitHub `RoboDojo-Benchmark/RoboDojo`，经海外代理核对）9/12 起在取观测前调用 `obs_manager.render_for_capture()`（`env/camera_manager/capture/render_sync.py`：渲染到相机缓冲区为当前物理状态为止），入口 `main.py` 通过 `add_zero_delay_kit_args` 启用 `checkForHydraRenderComplete=1000`、`renderer/waitIdle`、`hydraEngine/waitIdle`——与我们外壳的三个参数相同。当前官方评测即同步渲染；InternW0-Δ 在旧版（画面晚一帧）数据上训练，榜单成绩应出自 9/12 前的不同步评测，在当前协议下掉分。我们的数据对齐（官方数据 `OFFICIAL_RGB_SHIFT`，教师 / 专家同步录制）对当前协议正确，`iw0_memopen_v1` 不需改动。
- 帧时序核对（`official_t2/lag_corr.py`：每步画面变化量与关节变化量的滞后相关峰）：同步录制的教师 / 专家峰在 0，RDJ 官方旧版数据峰在 +1（画面晚一帧）。
- 上游 9/8 之后其余影响评测的改动只有两处指令：stack_blocks_by_language（"order ofblue" 补空格，9/16）、organize_table（指令改写，9/19）。本地两个任务文件已改为与上游逐字相同；`assemble_dataset.py` 组装时把教师 / 专家录制里的旧 stack 指令补上空格。
- 不同步模式下 Isaac 客户端随机卡在 `Resetting all environments`（带录制时必现），本地评测保持同步（`METIS_KIT_RENDER_SYNC` 默认 1）。
- 21:09 t2 GPU 0–4 起 InternW0-Δ 当前协议基线：官方 deploy、同步、14 个 Memory + Open 任务 × 50 集（seed 0 布局 0–49，同官方 `eval_nums`），`iw0_teacher/baseline_sync50/`。我们的模型最终也按此协议评测；不同步的 cover_blocks 50 集对照（`nosync50_cover_blocks`）跑完作参考。

### 评测协议与数据污染（2026-10-05 22:30–23:00）

**用户定的原则**：先让本地 InternW0-Δ 接近官方榜单，再要求我们的模型在同一本地协议下优于它；不用单个中间 ckpt / 单次评测下结论（seed、推理步数、episode 都会带来波动），模型选择用多 ckpt、多 seed 在开发布局上比较。

**官方协议是每任务 150 集**：`scripts/internal/summarize_result.py` 要求每任务 seed 0/1/2 三格（42 任务 × 3 = 126 格，每格 50 集，`EXPECTED_SEEDS = [0, 1, 2]`）；榜单逐任务 SR 都是 1/150 的倍数（92.7 = 139/150、62.7 = 94/150、18.7 = 28/150、74.7 = 112/150）。此前「协议 = seed 0 布局 0–9、每任务 10 集」的假设不对。

**榜单出分时是不同步渲染**：cover_blocks seed 0 布局 0–49，官方 deploy：不同步 45/49（score 93.8），同步 30/48（score 72.1），榜单 92.7；general_pickup 不同步 19/50、同步 13/50，榜单 62.7（仍差，原因未查）。上游 9/12 起官方客户端改为同步（见上文），InternW0-Δ 的榜单分应出自之前的版本。**本地「榜单协议」定为：本地 9/8 快照（不同步、旧指令）+ seed 0/1/2 × 50 集 + 官方 deploy**；19:40–21:00 按上游改过的两处任务指令（stack_blocks_by_language、organize_table）与组装时的 stack 指令补空格均已撤回，保持 9/8 版本。

**数据污染**：seed 0/1/2 的布局 0–49 都是评测布局。教师数据（seed 0 布局 10–54、seed 1/2 全部）与特权专家数据（seed 1/2）都采在评测布局上；在这些数据上训练的模型在官方协议下的分数不可信（测试场景泄漏，专家更是用真值解了测试场景）。处理方向（用户待定）：写布局采样器（任务配置 `task/RoboDojo/config/<task>.yml` 已有类别、xlim/ylim、rotate、select_mode，环境有 `check_layout_stability`）在 seed ≥ 3 生成新布局，专家数据在新布局重采（专家代码不变；16 卡约 2 h）；cover / conveyor 先只用官方示教（各 97 条，InternW0-Δ 只用官方数据即得 92.7 / 92.0），不重采教师数据。每个 seed 的布局 50–54 不在评测内（每任务 15 个），作开发集。已采数据保留，仅作内部诊断。用户提出重采成本问题，方案与估算已给出，尚未拍板。

**榜单协议验证**：22:47 在 t2 / t3 共 16 卡（每卡 1 客户端，`IW0_STALL_SECONDS=900`，`run_iw0_campaign.sh` 新增该变量）启动 `iw0_teacher/board_protocol/seed{0,1,2}`（14 任务 × 50 集，不同步，官方 deploy，`--attach-base-port 35990`），22:58 按用户要求暂停，尚无完成的轮次。恢复：各机按 GPU 起官方服务（端口 35990 + gpu），再用同样参数启动 manager（已验收集保留，claims 自动释放）。

### 22:58 全部暂停

用户要求暂停开发机与 t1–t3 的全部任务、只留保活：
- t1：VLABench v4 停在 step 15800（最新完整 ckpt `step_0015775`），保活由训练脚本按原参数恢复。
- t3：`iw0_memopen_v1` 停在 step 1760（最新 ckpt `step_0001640`，训练数据含评测布局，不再续训）。
- t2 / t3：InternW0 官方服务、榜单协议 campaign、cover_blocks 同步 / 不同步对照全部停止。
- 开发机：专家录制转换循环停止（`dataset_v1/<task>/rollout_expert/`：general_pickup 78、stack_blocks_by_language 59、press_by_number 130）。
- d1 之前已清空；22:54 出现的 `ig_memprobe`、`ig_sim4d` 两个会话不是本工作流启动的，未动。

### 特权脚本专家（见 `2026-10-05-RoboDojo-特权脚本专家.md`）

- general_pickup 批量 43/53（81%，教师 15/54），stack_blocks_by_language smoke 3/5（教师 7/55）；录制与教师格式一致，`build_gt4d.py` 原样转换。seed 1/2 全部 110 布局批量采集中：开发机（general_pickup 四块、stack 前半段），t2 GPU 0/1（stack 后半段 `expert_stack_s1b/s2b`）。
- Memory 任务专家（swap_blocks、press_by_number → imitate_sorting_sequence、swap_T）在 t2 GPU 2/3 开发。
- 18:16 general_pickup 专家 seed 1 + 2 共 110 布局采完，成功 79（72%）。19:00 转换（`build_gt4d.py --tag expert`，seed 1 偏移 0、seed 2 偏移 10000）进 `dataset_v1/<task>/rollout_expert/`，0 错误：general_pickup 78、stack_blocks_by_language 46（采集仍在进行）、press_by_number 6（Memory 专家 smoke）。stack 采集结束后再转一轮（已转的跳过）。

### 本模型首评

- step 278（训练早期）在 t2 GPU 4–7 跑 14 任务 × 10 集，验证评测链路（`scripts/eval_ckpt.sh`：`export_bf16.py --builder iw0` → mem 后端 campaign，执行 10 步），输出 `iw0_memopen_v1/simeval/step278_e10/`。

### 机器分配（10:15 起）

| 机器 | 任务 |
|---|---|
| t1 `rack-wlf2-ge26-36`（6 卡） | VLABench v3 avg4 评测 ID / Cat |
| t2 `a800bcctest0108-bd`（8 卡，新） | VLABench v3 avg4 评测 CS / Ins / Tex，之后 VLABench 下一轮训练 |
| t3 `a800bcctest0125-bd`（8 卡，即 10-04 的 t2，pod 重启） | Open 补采 → RoboDojo `iw0_memopen_v1` 训练 |
| 开发机 / d1 | Open 补采、转换、组装；之后 RoboDojo 闭环评测 |

- 交接链：开发机 tmux `iw0_chain`（`scripts/after_collect_train.sh t3 t3 dev d1`）等三台机器的采集 manager 全部退出 → 停 `iw0_convert` 并完整转换一轮 → 组装（Open ×4）→ t3 tmux `iw0_train` 启动 8 卡训练。日志 `outputs/MetisWAM4D_260921/iw0_memopen_v1/chain.log`。
- 10:37 `/m2v_intern_v3`（剩 64 GB）与 `/m2v_intern2`（0）写满，暂停写入：保活、守护、各脚本拉起保活时的 pid / 日志改到 `/ytech_milm_intern/danglingwei/logs/`，脚本用 `/ytech_milm_intern/danglingwei/wangrunqi_nvml_busy.py`；守护换成 `/ytech_milm_intern/danglingwei/files/keepalive_guard.sh`。t1 的保活（`high8_t1`，MPS 客户端）与守护已按新路径重启，各机作业不再持有这两块盘上的写文件。代码、权重等只读引用不变。
- 训练与评测脚本的保活匹配改为同时认 `/m2v_intern_v3/...` 与 `/ytech_milm_intern/...` 两种路径（新机器的保活在后者）；t2 / t3 的保活 pid 都是 22，共享盘上的两个 pid 文件同号，`run_metis_simeval.sh` 匹配到第一个即停。

## 限制与风险

- 不接 VLM：InternW0 论文中 VLM 是最大单项消融，Open 的收益不确定；Memory 主要靠记忆帧与视频先验。
- 教师蒸馏：缺口任务的数据来自 InternW0-Δ 滚动，提分中有一部分是其能力的蒸馏。
- Open 任务训练后不再是零样本。
