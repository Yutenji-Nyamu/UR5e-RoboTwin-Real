# MetisWAM4D: Learning Where and When to Focus for Task-Adaptive 4D World-Action Modeling

> 在 Janus4D 基础上，进行完善，目标 CVPR

## 数据集

### 人类 ego 视频预训练

第一部分 KlingHumanEgo-2.5M-5000H 数据，5000 h 时长（磁盘目录沿用 `KlingHumanEgo20M/`）：

- 元信息：/ytech_face_algo_ssd/lxq/captioner-data/20260526_13370_1_caption_v4_ego_local.parquet

第二部分：

- Apple Ego-Dex: /m2v_intern/public_datasets/EgoDex
- EgoVid: /m2v_intern/public_datasets/ego4d/EgoVid/v2/video_540ss
- EgoVerse: 访问不了；/kling_test/gen3/public_datasets/egoverse

### 仿真 robotwin2, robodojo

> 暂时不打算用 0.3 桶的不理想 Mock 数据；

### 真机实验

等 chenyiteng 合作；目前真机设定是 14D 双臂夹爪，走的这个框架：https://github.com/Yutenji-Nyamu/UR5e-RoboTwin-Real;

## 模型实现

- track 4d 定义在：以 $t$ 时刻相机坐标系下的物点3D运动
- video, track4d, action 三专家 Mot 网络还是基础和核心。
    + video 去噪最慢，track4d 去噪居中，action 去噪最快；所以三者独立的 timestep 采样，并且要满足这个设定。也就是说训练过程要通过一定timestep schedual 调度设计，让 video 噪声大于 track4d 大于 action. 并且要采样一定的边缘 case, 也就是：clean action (timestep 调度噪声为 0) 配对 noisy track4d, noisy video. 以及 clean action+clean track4d 对应 noisy video. (这个异步调度理念可以参考 xwam). 并且推理过程也是，action 推理少数步就结束了，但 track 和 video 还没有去噪结束，类似这种设计。
    + 人类预训练数据跟机器数据的一个区别是：人类有 ego 自身运动，而桌面固定机械臂头部相机几乎不动。所以需要有解耦自运动的能力。所以可以让 track4d expert 额外承担预测相机的任务；
- 空间上的 Salient Interaction Attention 以及时间上的 Transition-Anchored Aggregation 这两个技术要在上面 MoT 的基础框架上融入设计。可能需要一些额外模块，但也可能仅仅一些 Loss就够了。不管怎么样要 work, 要 novel, 要 make sense, 不能 trival.
- 由于 video 与本体的 track+act 可能不同源，因此是不是需要加上 video progress 和本体 progress 的预测来缓解进度不同的问题？以及是不是要本体编码，才能让 track expert 和 action expert 知道当前视频意图对应的应该是什么本体的 track 和 action?

### 最终方案要点（2026-09-24，代码 `metiswam4d/`，细节见 `docs/2026-09-21-MetisWAM4D-模型与训练课程实现计划.md`）
- **三专家 MoT**：Video（Wan2.2-TI2V-5B，3072-d/30 层）、Track（1024-d/20 层，映射到 30 个 macro 层）、Action（1024-d/30 层），共享一个联合 softmax（24 头 × 128）。Video/Action 权重直接装 OpenWAM-Alpha，Track 装我们的 Fusion-v3。
- **异步噪声**：统一进度 $u$，$\sigma_m=\max(1-u/r_m,0)$，$r_A=0.25<r_T=0.75<r_V=1$；训练混合 = 轨迹 0.4 / 有序独立 0.4 / clean-action 0.15 / clean-action+clean-track 0.05；推理 16 轮，Action 第 4 轮出结果。
- **Track4D**：$D(u,t)=R_t^\top(X^w(t{+}\Delta)-X^w(t))$，$t$ 时刻相机系的轴、自运动补偿后的物点运动，静止背景为 0；μ-law RGB 编码经 Wan VAE。相机自运动由 Track 序列末尾的 $N_f$ 个 **camera token** 承担（9 维：平移/0.1 m + rot6d − I，静止相机 = 0），与位移场同一 $\sigma_T$ 时钟。
- **空间聚焦（CSIA）**：Track 隐状态子帧展开 → 位移头/角色头（辅助监督）→ 本体参考运动、耦合场 $c$、转变 $\Delta c$ → 对数显著度 $\phi$（含相机运动幅度输入）→ 显著度 $s=\mathrm{softmax}_{grid}(\log\rho_{obj}+\phi)$（每帧一个分布，$\phi$ 的常数偏移被精确抵消，不存在"整体关掉"的零梯度态）→ 作为 $\gamma\log(s+\varepsilon)$ 偏置加在 $K$ 个查询的空间注意力上（物体侧/本体侧各一路）。邻域核固定高斯。
- **时间聚焦（TAA）**：锚点 = 帧转变强度 $s_n=\sum_i\rho_{obj}\,e^{\phi-\max\phi}$（按帧均值归一后再加 $\varepsilon$，尺度无关）的分位点（$K$ 个，无可学偏移），固定高斯时间窗（宽 0.15）对每个查询沿 $N_f$ 聚合。可学偏移/自适应窗仅作消融。
- **Action 读世界**：$K=32$/模态的紧凑 token，经**无门残差交叉注意力**进入 Action（O 小 std 初始化；零门在动作损失接近地板时是死锁）；dense 未来读取按 `dense_to_compact_steps` 退火后完全屏蔽。训练时对 Action 按样本屏蔽任务文本 / 本体 token（各 p=0.3），迫使其经世界接口取信息；监测 `focus/read_ratio` 与屏蔽/未屏蔽样本的动作损失差。
- **进度头**：`progress_video` / `progress_body`（层 0 干净 token + 文本，body 加本体状态 → 当前帧/episode 长度），MSE 0.05；预测值 $(p_V,p_E,p_V-p_E)$ 进入读取接口条件，为不同源的视频流与本体流提供共享相位。
- **损失**：video 1.0、track 0.2（本体/物体/背景分区归一）、camera 0.1、action 1.0、unfold 0.1、role 0.1、progress 0.05、condition 0.01。
- **训练**：HSDP（节点内分片、跨节点复制）16 卡，bs 8 × accum 1 = 128 窗口/步，~7.5 s/步；LR video 3e-7 / track 3e-6 / action 3e-6 / focus 1e-4；每小时 DCP ckpt（保留 5）+ 可视化（帧带、动作曲线、聚焦面板）+ TensorBoard；RT2 原始 episode 在线 VAE 编码，坏样本记录并跳过。



## 数据预处理

session: 35a3dedd-82de-4dcf-8f88-b76a68541095（2026-09-21 ~ 09-24）

新的 uvd displacement 表示：e7f14f8e-726f-45de-aa7e-b04366528513

IG10K 机器数据处理：eea77277-5839-44a0-b5fa-e71a4f80d56e

**原则：已有的能用就用，不能用的才重做；训练只读一份数据；每个环节先出 demo 再批量。**

### 最终产物

| 数据集 | 用途 | 训练根目录 / 索引 | 规模 |
|---|---|---|---|
| IG10K human | video + Track4D（物体 + 手）预训练主数据 | `/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/human/` + `manifest.parquet` | 4056 episode / 457,295 帧 / 4.23 h；标注 36 GB |
| IG10K robot | video + Track4D（物体 + 机械臂）+ action，第二阶段 mid-train 数据（`docs/2026-09-24-IG10K机器人子集预处理.md`） | `/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/robot/` + `manifest.parquet`（多 `data_dir`、`action_columns` 列） | 210 目录 / 10,500 episode / 3,679,424 帧 / 34.1 h；臂 mask 全覆盖（自带实例 id + 臂 id 200，或 SAM 双通道）；固定机位 |
| Kling `hi` 手部子集 | video + 手部 Track4D | `KlingHumanEgo20M/KlingHumanEgo20M_30fps_hi_handtrack_subset.parquet` | 145 万 clip / 2859 h |
| Kling `hi` 全量 | 仅 video（ego 视觉先验 + 语言对齐） | `KlingHumanEgo20M/KlingHumanEgo20M_30fps_hi_with_hands.parquet` | 341 万 clip / 7574 h / 3.17 TB |

**训练可用量汇总**：video 侧 341.5 万 clip + 4056 episode ≈ **7,578 h**；track 侧 145.1 万 clip + 4056 episode ≈ **2,863 h**（Kling 手部子集 2,859 h + IG10K 4.23 h）。IG10K 是唯一同时有物体 + 手部稠密 Track4D 的部分（3,256 episode 带手部 track；`_levels` 800 episode 无手部 track，其中 400 个物体 mask 偏弱待修）。

Track4D 统一定义：**相机坐标系下、源帧 t 到 t+4 的逐像素 3D 位移（米）**，stride-4 四相位合起来覆盖每一帧；静止 = 零位移，未跟踪区域 = 无效。渲染成视频时沿用机器人侧冻结编码（`Track3DCodec` mu=31，`scale_xyz = (0.022, 0.0135, 0.0165)` m，yuv444p crf10，零位移 128 灰、无效 0 黑），trainer 校验字段见 `training/raw_dataset.py`。

### 训练分辨率约定

- **训练分辨率 512×288**（16:9）。Kling 源 88% 本来就是 512×288；IG10K ego 由 1280×720 精确 2.5 倍下采样。
- 预处理只把短边归一到 288、保持宽高比，不做 crop/pad，留给 dataloader；换 chunk size / VAE stride 不用重做数据。
- Kling 中非 16:9 样本（竖屏 288×512 占 4.6%）在 dataloader 里按 `1.5 <= w/h <= 2.0` 过滤，保留 90.3%；不想丢竖屏就走 aspect-ratio bucketing。

### IG10K human

源：`imitator_human_v1`（53 目录，80% 帧）+ `imitator_human_v1_levels`（16 目录，20% 帧，为 5 个任务按 L1/L2/L3 单独录的人手示教，物体随级别变化）。两子集自带的标注**恰好互补**，决定了各自的链路：

| | `imitator_human_v1` | `imitator_human_v1_levels` |
|---|---|---|
| ego 相机键 | `zed2i` | `zed` |
| ego mask | ✅ 自带实例 id 流 + `mask_labels` | ❌ → SAM 3.1（prompt 由 Qwen3.5 从中文任务描述生成） |
| 手 | ✅ 自带 MANO（40 列） | ❌ → SAM 手臂二值通道，无手部 Track4D |
| ego 深度 | ❌（只有 exo 有）→ DA3 | 自带流编码未文档化，统一用 DA3 |

每个 `{subset}/{task_dir}/` 下：

| 文件 | 内容 | 生成脚本 |
|---|---|---|
| `videos/observation.images.*/` | 288p RGB（ego 512×288，exo 384×288），h264 crf18；LeRobot 时间轴不变，帧数逐文件校验 | `ig10k_human_resize.py` |
| `data/`, `meta/` | LeRobot 原表（MANO 参数、episode 时间戳）；info.json 已改 shape、去掉原始 depth/mask feature | 同上 |
| `depth/ep_*.mkv` + `depth_index.json` | DA3（da3nested-v11，单目/any-view，bf16，8 视图窗口）米制深度，12-bit 量化 + FFV1 无损（46.6 kB/帧） | `ig10k_human_depth_masks.py` |
| `masks.h5` | v1：uint8 实例 id 图 + `instance_labels`；levels：packbits 通道 `arm_and_hand,manipulated_objects`。`channels` 属性写明格式 | 同上 |
| `track4d.h5` | 物体 Track4D：RAFT stride-4 前后向光流 → DA3 深度反投影 → 相机系位移；有效性 = 双端深度有效 & FB<1px & 同类足迹。K 取 DA3 相机头逐目录中值（数据集无标定） | `ig10k_human_track4d.py` |
| `hand_track4d.h5`（53 目录） | 手部 Track4D：自带 MANO → smplx 网格 → 顶点跨帧位移 → pytorch3d 光栅化重心插值；**绝对深度锚到 DA3**，与物体同 K 同坐标系 | `ig10k_hand_track4d.py` |
| `track_rgb/ep_*.mp4` + `.codec.json` | trainer 直接读的冻结编码视频（物体 + 手合成）；**目前只渲了 8 个目录**，全量待启动 | `ig10k_render_track_rgb.py` |

**读数据一律走 `scripts/data_prep/ig10k_anno_reader.py`**：`load()` 返回 depth / objects / ids / arm / labels / mano，`load_track4d()` 返回物体 + 手合成的位移场（`hand` 掩码标出手像素），`manifest` 生成全集索引。

全量校验（69 目录各抽 1 episode，0 问题）：物体有效像素中值 3260/对、|Δ| 4.0 mm；手 1953/对、|Δ| 10.0 mm。抽检视频：`human/review_track4d/`（8 条 RGB|track 并排 + 拼图）。

必须知道的约定：

- **MANO 左手**：HaMeR 系对左手是翻转图像跑右手模型，存的参数要用 `MANO_RIGHT` + 原样参数 + 顶点 x 镜像（对 `pred_keypoints_3d` 验证到 0.0 mm）；smplx 一律 `flat_hand_mean=True`。
- **HaMeR 相机**：`pred_cam_t_full` 配的是全图焦距 `5000/256×1280 = 25000`，不是存的 5000（用 5000 手会缩小 16 倍）；其 Z（~31 m）是裁剪尺度伪值，不能当深度，所以手的绝对深度锚到 DA3。
- **SAM 3.1（levels）**：手臂用单 prompt `human arm` 逐帧检测（与跟踪版 IoU 0.976）；物体必须跟踪（只检测丢 45% 帧），先在 16 帧上筛掉本 episode 不出现的 prompt，再从首次出现帧双向传播，失败回退第 0 帧，单个 prompt 失败不拖垮目录。`GROUNDING_BATCH=4`、`SAM_MAX_OBJECTS=8` 控显存。
- **每个 episode 单独重置跟踪**，绝不跨 LeRobot episode 边界传播。
- 原始 IG-10K 的 depth/mask 流不在训练目录里（未文档化多通道深度打包、实例 id 掩码视频，播放器里看是"很近的深度"和"全绿"，属正常），留在 `IG-10K-Dataset/` 原处。

### KlingHumanEgo-2.5M-5000H

- **分桶**：按源 parquet 的 `ego_mani_score` 0.5 分 `hi`（341 万 clip / 7574 h，手部操作密集）与 `lo`（1381 万 clip，开车/走路等，视频可用但操作语义弱）。**决策：只做 `hi`；`lo` 暂停**，已完成的 453/5192 shard 保留（`lo_partial.parquet`，可按锁续跑）。
- **视频**：`-vf fps=30` 按时间戳补/丢帧（总时长不变），libx264 crf18，去音轨，本就 30/1 CFR 的 `-c copy`；不 resize（源 288p）。打 tar shard（`hi/videos_30fps/job-*/shard-*.tar`）避免 Ceph 上千万小文件，parquet 里 `video_path/offset/size` 直接定位。5192/5192 shard，0 失败。脚本 `kling_ego_30fps.py`。
- **Track4D 只做手**：逐像素 Track4D 8 亿帧不可行（实测 1.45 s/帧 = 190 GPU·天）。`hi` 上 WiLoR 覆盖 100%，其 npz 没存 MANO 姿态但 `joints_3d` 就是 MANO 21 关节且有 `betas`，**用解析 IK 反解**（手腕 Kabsch + 逐指两向量对齐沿链组合，0.5 mm；Adam 30 步打磨到 0.4 mm），前向只算 16 关节 + 5 指尖顶点，整 shard 批量，37.5 clip/s/卡。脚本 `kling_hand_mano_track.py`（check / fit / status）。
- **产物是稀疏拟合 + 在线渲染**：`mano_hi/job-*/shard-*.tar` 每 clip 一个 npz（θ48、β10、弱透视相机、frame_index、track_id、is_right、root_xy、rmse），25 kB/clip、全量约 90 GB。渲染 `render_clip()`：smplx 前向 → 778 顶点跨帧位移 → pytorch3d 光栅化 → 编码；一个 clip <1 s，放 dataloader 里做，不烘 1.2 TB 几乎全黑的视频。
- **可用子集**：WiLoR 有手的帧占比中值仅 0.39（这是硬限制）。`hand_det_per_frame ≥ 0.51`（按 2000 clip 校准 ≈ 真实覆盖率 ≥ 0.5）得 **145 万 clip / 2859 h** 作 video+track 样本，其余只作 video。抽检视频：`KlingHumanEgo20M/review_hand_track/`（60 条 + 拼图）。
- 必须知道的约定：位移 = 手内关节 XYZ 运动 + 手根 XY 平移，**手根 Z 置零**（`camera_translation` 的 Z 是弱透视伪深度，p99 |dZ| 2.65 m 纯框抖动）；21 关节是 OpenPose 序，指尖顶点 744/320/443/554/671；每 clip 每手 betas 取中值；帧对齐 `track 帧 k ← 源帧 round(k·src_fps/30)`；无 `track_id` 的旧 npz 用 `is_right` 兜底；`joints_2d` 已在 512×288 像素系，不需要 camera npz。

### 环境与运行约束

- 机器离网，只能用本地权重与代码。**GPU 脚本必须用 `/usr/bin/python3.10`**（vendored `pycocotools._mask` 是 cpython-310；DA3 依赖 `moviepy.editor` 只在 3.10 的 moviepy 1.0.3 有；smplx / pytorch3d 也装在这里）。
- 模型路径沿用 JanusTrack4d imperfect 数据那一套：DA3 代码 `.../m2v_intern2_danglingwei/codes/DepthAnything3_260406/src` + 权重 `.../files/depth_study_models/da3nested-v11`；SAM 3.1 `.../files/sam3_agibot` + `sam31_dependencies` + `depth_study_models/sam3.1/sam3.1_multiplex.pt`；RAFT `depth_study_models/raft_large.pth`；MANO `model_zoos/mano_v1_2/models`。`m2v_intern_public_models` 里的 sam3.1 / DA3NESTED 副本缺代码，不用。
- CPU 型作业（ffmpeg 重编码）和 GPU 型作业不要同机：同机时 load 227/116，GPU 吞吐腰斩。GPU 型作业每卡 2 个 worker 可把利用率顶满，配看门狗（`watchdog_ig10k.sh`）防 worker 静默退出后机器被回收。所有批处理都是"目录/shard 级 mkdir 锁 + 可续跑"，中断后清理无主锁即可继续。

### 已知限制与待办

1. **levels 物体 mask 偏弱**：16 目录中 H10_L2/L3、H13_L1/L3 大量 episode 物体通道为空，H15 四个只有 200 px 级。已定位为 Qwen 生成的 prompt 用词与 SAM 3 词表不合（`bag of chips` 0 命中，`plastic bag` 17/17 帧全中）。修法：prompt 加材质/形状泛词、筛选阈值 0.3→0.15，重跑这 8 个目录的 mask + track4d（约 1 h）。**待拍板**。
2. `track_rgb/` 全量渲染未启动（只有 8 个目录），trainer 读 IG10K 前要跑 `ig10k_render_track_rgb.py run`（CPU，约 20 min）。
3. 物体 Track4D 在图像边缘偶有 RAFT 光流伪影（抽检 H58 右上角细条），未加边缘裁剪。
4. Kling 手部 Track4D 时间上稀疏（子集内仍有约一半帧无手），训练时黑色区域要在 loss 里 mask 掉，无手帧按"track 缺省"处理。
5. EgoVid（`/m2v_intern/public_datasets/ego4d/EgoVid/v2`，4.9M 文件 / 4.7 TB，自带 HaWoR 手部 MANO + 相机轨迹）可复用 Kling 的 MANO 链路，尚未体检；任何递归扫描需先审批（>1M 文件）。

## 机理探查实验

session: 823591ed-3e6b-477b-911c-ca7e77dbfb4b

- 创新点一的探查主要是证明：没有 3d 4d 几何信息的话就抓不准 或者 这类任务成功率低。
- 创新点二主要是探查现有的 wam action 基本上没法很好地 attention 到关键区位；
- 创新点三主要证明 zero-wam 等方法哪怕没有 Attention 到原视频也可以完成任务，呈现出一种 acton expert 过拟合的状态。


> 2026-09-22 规划（v2）。定位：**纯推理探查**，在 S1/S2 训练出来之前，用现有 baseline 把 MetisWAM4D 三个贡献点各自的"问题存在"证明出来，产出 Experiments 的 motivation / analysis 素材。不用分叉数据（那不是本文贡献）。
>
> 三个贡献点各对应一条待证命题：
> **(a) Track4D 表征与模型 → 现有表征不足**：像素中心的未来预测（Alpha 的 Video latent）不承载本体—物体的度量交互关系，也预测不好物体；Track4D 承载。
> **(b) 时空聚焦（CSIA/TAA）→ 现有读取冗余且不聚焦**：稠密未来 token 高度冗余，真正影响动作的只是小部分且有耦合结构；Action 对世界的注意力/归因既弥散又不落在交互区。
> **(c) 意图与本体解耦 → baseline 里 video 与 action 是松耦合的**：替换/打乱未来 video，action 基本不变；action 主要由本体状态+任务先验决定。所以 video 通道本来就只承担"意图"信息，把它与 track+action 的本体时间轴解开不会损失 baseline 已在用的东西。
>
> 每条命题都要在**两个 baseline 上做同一套探查**：OpenWAM-Alpha Sim-RoboTwin-Full（Video+Action，像素中心 WAM 代表）与 JanusAct4D-RT2 `savestep_19045`（Video+Track4D+Action 稠密双向读取，RT2 95.5%，是 $M_{\mathrm{full}}$ 的代理；合并权重 `outputs/JanusAct4D_imperfect/rt2_v1/simeval_19045/checkpoint/mp_rank_00_model_states.pt`）。两者对比本身就回答"Track4D 带来了什么、还缺什么"。
>
> 机器：本机 2×A800 80GB；RoboTwin 仿真在 `/usr/bin/python3.10`（sapien 3.0.0b1），闭环评测复用 `janusact4d_rt2imperfect_v1/evaluation/`（campaign/worker/policy），把 8 卡 32 并行缩到 2 卡 8 并行。探查子集固定为 **10 个代表任务 × clean/random × 20 个固定初态 = 400 条/条件**（任务按交互类型选：抓放、堆叠、关节体、按压、双臂交接、需读指令的 a2b/ranking 类），开环筛选用 batch_v1 的验证窗口（约 500 个，按任务分层，模型未见过的 episode 优先；若训练用了全部 episode，则开环只用作相对比较，绝对结论以闭环新初态为准）。
>
> 数据事实：batch_v1 `source.hdf5` 有 GT depth、robot/object 语义 mask、相机内外参、末端位姿与夹爪开度，**没有物体位姿**；物体三维运动用 GT depth + mask 反投影得到（质心/刚体拟合），事件（接触建立 / 支撑转移 / 释放）由夹爪开度变化 + 物体是否与末端共动 + 物体离开支撑面导出；**交互区** = latent 网格上本体与物体三维距离 < 3 cm 的 token 及其邻域。batch_v1 的 Track4D 物体侧是 RAFT+GT 深度，不是纯真值，凡是拿它算耦合场的地方都要写明。

### P0 共享基础设施（约 2 天）

- 事件与交互区真值：`scripts/probe/gt_events.py`，从 `source.hdf5` 逐 episode 导出物体三维轨迹、事件帧、交互区掩码（latent 网格、33 帧 / stride-4 窗口对齐）、耦合场 $c,\Delta c,\pi$ 与帧级 $s_n$（用 `metiswam4d/focus/coupling.py`，$\phi$ 固定为 $|\Delta c|$）。顺带产出 §3.2 要的窗口内事件统计和角色覆盖率。
- 双模型探查驱动：`scripts/probe/harness.py`，统一装载 Alpha / Janus，在联合注意力的 K/V 接口上对 **Action query 可见的未来世界 key** 做干预（置零 / 换样本 / 换任务 / 时间打乱 / 静止化 / 按 token 子集保留 / 均匀池化），并可导出显式 softmax 的注意力权重与输入梯度。干预只作用于 Action 的读取，不改 Video/Track 自身的去噪，保证"世界预测不变、只改 Action 看到什么"。
- 开环指标：动作块 EEF 位置/姿态误差、夹爪开合时序误差、相对未干预输出的动作偏移；闭环指标：SR、成功步数。
- 闭环 harness 在本机跑通 Alpha 与 Janus 各 1 条 smoke，确认 2 卡吞吐（预计 400 条 ≈ 3–4 h）。

### P1 现有表征不足（命题 a）

- **P1.1 线性可解码性**：冻结模型，缓存各读取层上未来世界 token 的隐状态（Alpha video / Janus video / Janus track），用岭回归 + 5 折按 episode 划分，解码：物体米制位移、本体—物体相对位移（耦合向量 $r$）、耦合状态 $c$、当前是否处于接触、距下一事件的帧数。特征 PCA 到同维再比较；报告 $R^2$ / AUROC，附打乱标签基线。预期：像素 latent 能解码本体运动（FK 决定）但解不出物体位移与相对运动；Track 特征两者都能。这是"表征不足"最标准的证据形式。
- **P1.2 分区域的世界预测质量**：在验证窗口上按 GT mask 把预测未来分成**本体区 / 物体区 / 背景区**分别打分：Alpha 解码 RGB 的掩码内 PSNR/LPIPS、DA3 深度误差、RAFT 提取的物体运动与 GT 物体三维运动的误差；Janus 的 Track 米制 EPE 按 Body/Object 分列。预期像素 WAM 本体区好、物体区差（物体被抓起后的运动、放置后的位置常常糊掉或不动），即像素表征对交互后的物体状态建模弱；Track 模型物体 EPE 明显更低。这是命题 a 最直接的量化。
- **P1.3 小位移分辨率（VAE oracle）**：真值 Track4D → μ-law → Wan VAE 往返 → 米制 EPE（Body/Object 分列、按位移幅度分桶），同时测 RGB 路径下 5 mm / 1 cm 物体位移在 latent 差分中的可检出性。它既是所有 Track 误差的下界，也说明接触前后毫米级关系变化在哪种表征里才"看得见"。
- **P1.4 预测质量与控制质量的解耦**：在闭环子集上逐 episode 记录 Alpha/Janus 的世界预测误差（视频 PSNR、Track EPE）与结果（成功与否、动作误差），算相关；接近零说明"世界预测得好"并没有转化为"控制得好"，为 P3 铺垫。

### P2 冗余且不聚焦（命题 b）

- **P2.1 表征级冗余**：各读取层未来世界 token 的有效秩 / 参与比、相邻 token 与相邻帧的余弦相似度、PCA 保留 k 个主成分后重建 K/V 引起的动作偏移；给出"$N$ 个 token 里真正独立的自由度约几个"。
- **P2.2 功能级冗余（因果版，本节主图）**：推理时对 Action 可见的未来世界 token 只保留一部分：随机 p%、只留交互区、只留耦合显著度 top-k、只留运动幅度 top-k、只留事件邻近帧、只留均匀间隔帧；保留率取 100 / 20 / 5 / 1%。开环全部条件，闭环跑 100% / 随机 5% / 显著度 5% / 运动幅度 5% / 交互区 四组。预期："5% 显著度 token ≈ 全量、随机 5% 明显掉、运动幅度居中"——同时证明冗余和"有用子集有耦合结构"，是 CSIA 的直接动机；帧维度的同款结果是 TAA 的动机。
- **P2.3 注意力与归因是否聚焦**：Action query 对未来世界 key 的注意力：归一化熵、交互区提升比（质量/面积）、逐帧质量曲线与事件帧的距离、注意力汇聚（attention sink）占比、按层与去噪步分解；同一套指标再用输入梯度 × 输入 / Integrated Gradients 做一遍，避免"注意力≠解释"的质疑；最后加一个因果检验：遮掉 top 注意力 token vs 遮掉 top 显著度 token vs 随机遮同样数量，看哪种对动作影响最大。若注意力弥散、归因弥散、且遮 top 注意力不比随机更伤，就是"不聚焦"的三重证据。Alpha vs Janus 对比回答"加了 Track 之后注意力有没有自动变聚焦"——预期没有，这正是为什么聚焦要靠 CSIA/TAA 显式给出。
- **P2.4 定性图**：抓取瞬间 / 放置瞬间的注意力图与归因图叠在预测未来上，两模型并排，配 GT 交互区轮廓。

### P3 video 与 action 松耦合（命题 c）

- **P3.1 未来 video 替换（开环 + 闭环，本节主图）**：Action 可见的未来 Video key 换成：同任务另一 episode、不同任务、静止化（重复当前帧）、时间打乱、时间平移 ±k 帧、置零；Janus 另做"只换 Track / 只换 Video"。闭环跑 6 个条件。若 SR 几乎不掉，说明 baseline 的 action 不读 video——世界模型是装饰。时间平移 / 打乱不变 → action 没有利用"video 与 action 同时间轴"这一假设，正好说明 video 通道可以脱离本体时间轴、只作意图条件，这就是解耦的依据。
- **P3.2 video 对 action 的边际贡献**：与 P3.1 配套，把 Action 可见的世界 key 全部去掉（等价于 Action-only）与只留 Track（Janus）对比，得到 "video / track / 二者都无" 三档 SR 与开环误差：量化 video 通道在 baseline 里实际贡献了多少控制信息、Track 通道又贡献了多少（呼应 §7.1 表示角色消融，但这里是推理侧、零训练）。
- **P3.3 action 主要由什么决定**：(i) **观测换样本**：当前观测帧换成同任务另一初态的帧、本体状态保持，看首个动作块跟着观测走还是跟着本体走（闭环下等价于"看错场景还能不能成功"）；(ii) **指令替换**：在同场景对 a2b_left/right、blocks_ranking、pick_diverse_bottles、handover 这类指令决定目标的任务换成另一合法指令，闭环按新指令判成功——跟场景先验走就失败，说明意图通道没被真正使用；(iii) **kNN 可解释度**：对新初态，用初始本体状态 + 物体三维位置检索训练集最近邻 demo，其动作块对策略输出的解释方差 $R^2$。三项合起来说明 baseline 的 action 主要由本体状态与任务先验决定，意图信息（video、指令）利用不足；正文措辞用"利用不足"，不用"记忆"。

### 顺序、预算与产物

1. P0（2 天）→ 2. P2.3 + P3.1 开环（1 天，最快出结论，决定后面闭环跑哪些条件）→ 3. P2.2 + P1.2 开环（1–2 天）→ 4. 闭环：P3.1 六条件、P3.2 三档、P2.2 四条件、P3.3(i)(ii)，两模型，约 20 组 × 400 条（2 卡约 3–4 天，可与开环穿插）→ 5. P1.1 / P1.3 / P2.1 / P3.3(iii)（CPU + 1 卡，穿插）→ 6. 图表：Figure 2 = P2.2 保留率曲线 + P3.1 替换条件 SR；P1.2 / P2.3 进 Analysis 小节。
代码放 `scripts/probe/`，结果放 `outputs/MetisWAM4D_260921/probes/<name>/{metrics.json,figs/}`，每项在 `docs/` 记一段结论与限制。所有探查只改 Action 的读取输入，不改模型权重；闭环协议与前身 simeval 完全一致（seed、执行步数、去噪步数、指令集），否则数字不可比。

### 不在本轮做

- 需要 $M_{\mathrm{focus}}$ 的项（TAA 锚点精度、CSIA 质心随指令变化、等预算读取对照、多速率早停）：等 S2；P0 的 harness 与真值直接复用。
- 人类 ego 数据上的探查：Kling / IG-10K human 尚无 Track4D。
- RoboDojo 侧复核：等 RT2 上结论稳定再决定是否值得在 Isaac 上重跑一遍 P2.2 / P3.1。

## 训练课程

注意利用预训练模型先验，并通过合适的 LR 避免冲垮先验；

- 第一步：预训练
    + 先用 KlingHumanEgo-2.5M-5000H 数据与 IG10KSelf 中的人类示教数据混合训练；只有video,track4d 没有 action;
    + video expert 可以用 OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model 初始化；track expert 可以插值初始化并冻结 video expert 预热 track expert；
    + 这一步虽然没有 action expert 但上面设计的 多专家、异步噪声调度、空间显著交互注意力、时间变化瞄定的整合也应该有；
    + 这一步要考虑预训练过程的 chunk size 是不是要和后面的机器人数据的 chunk size 统一；
- 第二步：加入 action expert 做 Mid train; 用 IG-10KSelf 中的机器人数据；但需要考虑的问题是：设置统一的 action expert 接口，便于下游在 robotwin2 数据后训练，以及 robodojo 数据后训练，以及在真机数据后训练（目前真机设定是 14D 双臂夹爪，走的这个框架：https://github.com/Yutenji-Nyamu/UR5e-RoboTwin-Real）；最好用统一的动作表示和维度吧？action expert 你觉得用 openwam-alpha 系列哪个 ckpt: https://huggingface.co/collections/OpenWAM/openwam-alpha;
- 第三步并行后训练；
    + robotwin2 数据； 
    + robodojo 数据；
    + 真机数据；
    + 扩展应用：用 IG10KCross，解耦 video 与 track+action;


### 预训练：

session: 10a598a5-3b63-46c4-945d-36513edc723f
klinghuman, IG10K human+robot: ef73aad7-ffd7-4912-8969-454dfe0733c4

实现完成、等双节点 16 卡（2026-09-24，细节见 `docs/2026-09-24-人类数据预训练-实现记录.md`）：

- **数据**：Kling 手部子集（video + 手部 Track4D）0.5 / IG-10K human（video + 物体 + 手部 Track4D）0.3 / Kling 全量 video-only 0.2；batch 模态同构、组序列跨 rank 同步；IG-10K 每目录留 2 集做验证（每 1000 步）。统一 512×288 单视角（token 9×16）、33 帧 / stride 4，与机器人阶段同窗口；跨阶段网格差异由 Wan 3D RoPE 承担。
- **初始化**：Video ← Alpha Foundation；Track + reader + 进度头 ← `rt2_direct_open` step 11028（`init.initialize_from_prefixes`）。Video 冻结 2000 步再 2000 步升到 3e-7；Track 3e-6，focus 1e-4；50k 步；两专家异步噪声（$r_T=0.75<r_V=1$，clean-track 边缘态 0.2）；CSIA/TAA 经无门残差回读由 Track FM 损失训练。
- **自运动**：Kling 外参（world→cam，光流核对）补偿手部位移并监督相机 token 旋转（平移尺度未核实，权重 0）；IG-10K 固定机位，相机 token 标签为单位阵。Kling 无深度条件 → Track 专家学习到的缺失替身 token。
- **在线编码**：Wan VAE + UMT5 在训练进程内编码，不落 latent / 文本缓存。
- 启动：`NODES="tX tY" bash scripts/human_pretrain/launch_2node.sh`；抽检：`scripts/human_pretrain/demo_windows.py`；输出 `/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/stage1_pretrain_human/`。

### 第二步：联合预训练（Kling + IG-10K human + IG-10K robot，三专家）

2026-09-27 22:56 在 t3 + t4 启动，细节见 `docs/2026-09-27-联合预训练-人类机器人三专家-实现与运行记录.md`：

- **数据**：Kling 0.4（手部 0.28 / video-only 0.12）: IG-10K 0.6（human 0.24 / robot 0.36）；批次按源同构，robot 批次带 action + proprio，其余只有 video (+ track)。
- **动作**：Realman `eepos_gripper` 14 维 → Alpha EEF20（Euler xyz → rot6d）→ 统一 80 维槽 0-9 / 34-43，min-max（`robot/action_stats_eef20.json`）。
- **模型**：embodiment token 进三专家 context；progress 三头（video / body / action）；camera token 沿用；Action `compact` 读取从 dense 退火 6000 步；`feedback_to_track` 保留。
- **初始化**：Video / Track / reader ← `stage1_pretrain_human_uvd` step 21609；Action + proprio encoder ← Alpha Foundation。
- **训练**：bs 16 × 16 卡 = 256 窗口/步，重算，~16 s/步，30k 步；Video 冻 1000 + 升 2000 → 3e-7，Action 冻 500 + 升 1500 → 3e-6，Track 3e-6，focus 1e-4；track 权重 1.0。
- 输出 `outputs/MetisWAM4D_260921/stage2_pretrain_joint/`，守护 `guard.sh`，配置 `configs/stage2_pretrain_joint.yaml`。

### rt2 训练

session: b462fd18-f722-4fb9-b4ab-1f3844b1e284

2026-09-29：`rt2_direct_v2_uvd` 在 30804 步停止（Action `read_ratio` 始终 ~1%，异步调度下 Action 去噪时世界接近纯噪声），改为 `rt2_direct_v3_coupled` 续训：Track 与 Action **共钟**（r_T = r_A = 0.5，u ∈ [0.5, 1] 为 clean action + track 条件下的视频生成，推理 20 轮、第 10 轮出动作）；dense 读取永久保留（`dense_read_floor 0`，≥ 旧版 triple expert），compact 读取为残差；`ReadActionHead` 探针损失防接口死掉；LR video 5e-6 / track 1e-5 / action 2e-5。记录见 `docs/2026-09-23-RT2直接训练-运行记录.md`。

2026-09-30：v3 到 step 44949（t1 被回收）；闭环 SR 90.4% / 89.8%（33760 / 40472），`loss/action` 40k–45k 平台 0.0020。改 `rt2_direct_v4_anneal` 在 t3 续训 25k 步到 70000：各组 LR 从 v3 值余弦降到 1/10（新增 `training.lr_schedule_start`），`action_text` / `action_proprio` dropout 0.3 → 0.1，其余不变。

2026-10-02：RoboTwin 2.0（50 任务 × {clean, random} × 100 集，seen 指令 + SR93 归档 seed）最终结果：**avg5**（v3 step 41965–44949 五个 ckpt 均值，hanging_mug 执行 24 步）平均 **93.20%**（clean 93.62% / random 92.78%，9320/10000；10-04 补齐 handover_block 180 集），持平 JanusTrack4D-v3 93.10%。OpenWAM-Alpha 官方 ckpt 按官方协议（unseen 指令、官方 seed 序列）本地实测 92.33%（README 93.60%），两者口径不同。v4_anneal 的单 ckpt 53898（91.0%）与 avg5（92.4%）均未超过 v3 avg5。弱任务 place_can_basket 46%、hanging_mug 66%、open_microwave 76%。记录见 `docs/2026-09-29-RT2闭环仿真评测-实现与运行记录.md`，产物 `outputs/MetisWAM4D_260921/rt2_direct_v3_coupled/simeval_avg5_e24/`。

### robodojo 后训练

session: dbb49913-1d71-4d34-9665-6aa59c03532e

2026-09-28 22:04 在 t2（8 卡）启动 `stage3_robodojo`，细节见 `docs/2026-09-28-RoboDojo后训练-数据方案与RDJ数据集构建.md`：数据 `RDJ_MetisWAM4D`（机器人 FK 像素系 uvd track，物体未知不监督）；Video / Action ← Alpha RoboDojo，Track + 接口 ← `rt2_direct_v2_uvd` step 25003；动作为 Alpha RoboDojo 的**逐臂基座系** EEF20 + 其 min-max；128 窗口/步，50k 步，每小时 ckpt。输出 `outputs/MetisWAM4D_260921/stage3_robodojo/`。

2026-09-29 切共钟方案 `stage3_robodojo_v2_coupled`（与 RT2 v3 同配方，从 13407 续）；2026-09-30 到 step 27519 停止，`loss/action` 20k–27.5k 平台 0.0036。改 `stage3_robodojo_v3_anneal` 在 t4 续训 25k 步到 52500：LR 余弦降到 1/10、捷径 dropout 0.3 → 0.1，与 `rt2_direct_v4_anneal` 同配方。

### EBench 后训练

选榜依据见 `docs/2026-10-02-备选榜单调研-WAM适配性对比.md`：OpenWAM-Alpha-Sim-EBench 与本框架完全兼容，Test-Mini SR 49.4，已是第一。2026-10-03 08:43 在 t1 启动 `stage3_ebench_v1`：EBench-Dataset 26 个 bucket，经 OpenWAM 的读取器在线读取；Video / Action ← Alpha EBench，Track + 接口 ← RoboDojo v3 anneal step 37951；RoboDojo v6 配方，这一版没有 Track 目标；128 窗口/步，20k 步，每步约 5 s。环境、数据、训练和评测问题见 `docs/2026-10-02-EBench环境与评测-实现与运行记录.md`。

### VLABench 后训练

10 个 primitive 任务用 VLABench 自己的场景采样与专家技能重新生成（`scripts/vlabench/generate_4d.py`），带仿真真值的头相机深度、机器人 / 夹爪 / 运动物体角色和刚体 Track uvd；Video / Action ← OpenWAM-Alpha-Sim-VLABench（avg SR 58.9），Track + 接口 ← RT2 v4 anneal；RT2 v4 配方，`configs/stage3_vlabench_v1.yaml`。环境、数据、训练记录见 `docs/2026-10-03-VLABench-4D数据生成与训练记录.md`。

2026-10-04：OpenWAM-α VLABench 官方 ckpt 按官方协议（5 track × 10 任务 × 50 集）在 t1 复测，Avg SR / PS / IS = **57.6 / 66.1 / 65.7**（论文 58.9 / 67.2 / 63.5）；评测管理器 `metiswam4d/eval/vlabench_campaign.py`，产物 `outputs/MetisWAM4D_260921/vlabench_alpha_official/`。

### 仿真评测

robotwin2：227557dc-2042-4d95-9c60-5680a72f56fd

robodojo: 7a60cb3d-1bb7-405d-ac53-591474ec55bc
