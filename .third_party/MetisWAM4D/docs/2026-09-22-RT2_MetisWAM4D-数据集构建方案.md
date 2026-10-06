# RT2_MetisWAM4D：RoboTwin 2.0 干净 4D 数据集

> 目标目录 `/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/RT2_MetisWAM4D/`（ytech_milm_intern，剩余 115 TB）。
> 原则：能复用的软链接，缺的在 t4 重算；**所有几何量来自仿真真值，不用 RAFT / DA3 / SAM**。

## 1. 现有资产盘点

| 资产 | 位置 | 状态 | 处理 |
| --- | --- | --- | --- |
| 原始 episode HDF5：三视角 RGB（jpeg）、`front_camera`、**仿真深度**（float16 mm，240×320）、四相机内外参、`semantic_mask`[场景物体, 夹爪, 整机]、`joint_action`、`endpose` | `ytech_face_algo_ssd_danglingwei/datas/GeoRobotwin/<task>/{demo_clean_4d,demo_randomized_4d}/data/episodeN.hdf5`，50 任务 × (50 + 500) = 27,500 集 | 干净 | 软链接 `source.hdf5` |
| 回放种子与轨迹 | 同目录 `seed.txt`、`_traj_data/episodeN.pkl` | 干净 | 回放输入 |
| 逐像素 actor 实例 id（head/left/right，uint16）+ 机器人 id、任务物体 id、接触选中 id | `GeoRobotwin_JanusAct4D_Imperfect/batch_v1/<task>/<variant>/episodeN/task_instances.h5` | 干净（seed 回放渲染，前身已校验关节/相机/深度 P99<2 mm） | 软链接 `task_instances.h5` |
| 指令（每集 seen/unseen 池） | `GeoRobotwin_JanusAct4D_Imperfect/episode_instructions_sim_aligned.jsonl`（27,500 行） | 可用 | 软链接 |
| UMT5 文本缓存 | `GeoRobotwin_JanusAct4D_Imperfect/text_cache/embeddings.sqlite3` | 可用，覆盖率待核对，缺的补编码 | 软链接 + 增量 |
| Track4D（前身） | qpos 桶：仅机器人；0.3 桶：DA3+RAFT+SAM 估计 | **不用** | 重算 |
| 深度（前身 0.3 桶 `da3_depth.h5`） | — | 不用 | 用原始仿真深度 |

## 2. 缺失项与重算

唯一缺的是逐像素 3D 对应。刚体（机械臂各 link、夹爪、任务物体、可动 clutter；铰接物体按 link）满足

$$
X^{c_t}_{t+\Delta} = W_t^{-1}\,T_e(t+\Delta)\,T_e(t)^{-1}\,W_t\,X^{c_t}_t,\qquad D_t(u)=X^{c_t}_{t+\Delta}-X^{c_t}_t ,
$$

其中 $X^{c_t}_t$ 由仿真深度反投影，$e$ 是像素所属实体（`task_instances.h5`），$T_e$ 是实体位姿，$W_t$ 是头相机外参。因此**只需回放物理记录每帧所有实体（actor + articulation link）的位姿**，不必渲染。

- `scripts/data_prep/rt2/replay_poses.py`：按 seed + `_traj_data` 回放（前身 `robotwin_task_masks.py` 的做法），`_take_picture` 只记录位姿与关节；校验关节向量与源一致、实体 id 集合与 `task_instances.h5` 一致；输出 `poses.npz`。
- `scripts/data_prep/rt2/build_track4d.py`：由 `source.hdf5`（深度、内参、外参）+ `task_instances.h5` + `poses.npz` 计算 stride-4 位移 `delta_xyz_cam [T-4, 240, 320, 3]`（float16，米，头相机坐标）与角色图 `role [T, 240, 320]`（0 背景 / 1 机器人 / 2 物体），写 `track4d.h5`（gzip）。物体 = 任务实体 ∪ 任务中发生运动的 clutter；从未运动的 clutter 与桌面/墙为背景（位移 0）。
- 全局尺度：抽样统计三轴 P99.5 → `track_norm.json`；μ-law（μ=31）RGB 编码在 dataloader 内完成，不预存视频。
- 运行：t4 八卡（GPU 只给 SAPIEN 渲染系统占位，主要吃 CPU；t4 正在跑 Kling ffmpeg，进程数保守）；先 clean（2,500 集）后 randomized（25,000 集）。

## 2.1 实施记录（2026-09-22）

- 前身实例 id 只覆盖 clean 200/2500、randomized 2029/25000；其余集在回放中用前身的单次主采样 RT 着色器自行渲染头相机 `Segmentation`，并逐帧与源仿真深度比对（P99 误差 0.25 mm = float16 量化，超过 2 mm 即报错）。有前身文件的集直接软链接。
- 前身 RoboTwin 检出中的 curobo yml 写死了已删除目录的绝对路径，本仓库 `third_party/RoboTwin/` 是符号链接 overlay，只重写这两个 yml（`scripts/data_prep/rt2/make_robotwin_overlay.sh`）。
- 每进程按任务复用 env（规划器初始化约 2 分钟/任务只付一次）；稳态每集回放约 14–66 s（取决于 t4 CPU 争抢，t4 同时在跑 Kling ffmpeg）+ 构建约 11–22 s；`track4d.h5` 约 12 MB/集。
- 启动：`bash scripts/data_prep/rt2/launch_rt2_track4d.sh demo_clean_4d 16`（t4，tmux `rt2track_*`）；状态 `rt2_track4d.py status`；完成后 `index`、`norm`（全局 P99.5 与深度统计）。
- 首批 8 集统计：`scale_m` ≈ [0.091, 0.057, 0.054]（stride-4 位移 P99.5），头相机深度 0.23–1.03 m。clean 全部完成后需重算并冻结。

## 3. 训练侧

- `metiswam4d/data/rt2_episode.py`：读 `source.hdf5` 三视角 RGB → Alpha 的 L 形拼图（384×320：head 256×320 上，左右腕各 128×160 下）；头相机深度 → 米制 → 条件图；机器人|物体掩码 → 条件图；Track4D RGB（μ-law）；EEF20（endpose→pos3+rot6d+gripper，Alpha RoboTwin `normalization_stats.npy` min-max）→ 统一 80 维槽 0-9/34-43；文本从缓存取。
- 在线 VAE 编码（Wan2.2 VAE，diffusers `AutoencoderKLWan`，`(z-mean)/std`，与 Alpha 编码一致）在训练循环内做，避免 ~1 TB 的窗口级 latent 落盘。
- 初始化：Video/Action ← `OpenWAM-Alpha-Sim-RoboTwin-Full`（step 118655）；Track ← Fusion-v3 `step_30000` 的 `model.track_branch.*`（键名重映射：`expert.blocks.i.context_attn`→`blocks.i.cross_attn`，`norm2`→`norm3`，`expert.time_*`→`time_*`，`expert.final_modulation`→`head.modulation`，`head`→`head.head`；RoPE 分轴与频率、AdaLN 顺序、正弦时间嵌入均与 Wan 一致，已逐项核对）。
- 配置 `configs/rt2_direct.yaml`：三专家、compact + dense→compact 退火、异步噪声；在 t5+t6 拉起。
