# codex 交接提示词：RoboDojo 重点任务提分（2026-10-01 22:45）

在 t2 上 `tmux new -s codex`，`cd /m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921`，启动 codex 后整段粘贴。

---

你在训练机 t2（8×A800-SXM4-80GB，BCC pod，`a800bcctest0080-bd`）本机工作，项目根目录 `/m2v_intern_v3/danglingwei/codes/wam_proj/MetisWAM4D_260921`。先读 `.cursor/rules/` 下全部规则（00-workflow、10-cluster-ops、20-records-and-deliverables、30-training-runs、40-kuaishou-infra）并遵守。我接下来约 24 小时在路上无法回复：**本次任务授权你全程自主执行**，按下面的决策规则办，不要停下来等我确认；只有遇到会破坏数据 / 他人作业、或超出本文范围的事才停。

## 一、背景

- 模型 **MetisWAM4D**：三专家 MoT（Video = Wan2.2-TI2V-5B 骨干；Track4D = 头相机像素系位移 (Δu, Δv, Δd) 的流匹配专家；Action = EEF80 统一动作），Action 通过 reader 读取未来 Video/Track；Track 与 Action 共钟（20 轮采样、第 10 轮出动作）。代码包 `metiswam4d/`，训练入口与配置见 `configs/`、`plan.md`。
- RoboDojo 后训练谱系：`stage3_robodojo_v2_coupled`（到 step 27519）→ `stage3_robodojo_v3_anneal`（t4 续训到 37951 后停；现存完整 ckpt `step_0034962 / 35709 / 36456 / 37202 / 37951`，在 `/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/stage3_robodojo_v3_anneal/checkpoints/`）。训练数据 `RDJ_MetisWAM4D`（35 任务 × 100 集官方示教，Track 只含机器人 FK；数据文档 `docs/2026-09-28-RoboDojo后训练-数据方案与RDJ数据集构建.md`）。
- 闭环评测体系已建好并验证（`docs/2026-09-30-RoboDojo闭环仿真评测-实现与运行记录.md`，**必须先完整读完**）：本地缩减协议 42 任务 / 54 配置 / 420 条（官方 layout seed 集 0 的 layout 0–9），`metiswam4d/eval/rdj_campaign.py prepare/run/summary` + `scripts/eval/run_rdj_simeval.sh`，每卡一个策略服务（python3.10）+ N 个官方 Isaac 客户端（`/usr/local/robodojo-python-runtimes/.../python3.11`），训练—推理几何逐像素一致。
- 当前最好结果：**anneal 34962 + `hide_track`**（推理时对 Action 隐藏 Track 分支）= 46/420，Overall SR 10.75 / Score 17.14（`stage3_robodojo_v3_anneal/simeval_34962_hide/`，逐任务表 `paired_vs_baselines.md`）。对照：OpenWAM-α 本地 9.58 / 官方榜 11.92，JanusAct4D-RDJ 40k 10.75。榜单 SOTA WAM 22–31。
- 已排除的提分手段（都在噪声内）：换 ckpt、权重均值、cfg / 多采样 / 更多轮、执行 16 步 + chunk 集成。Open 维度 8 个任务不在训练集里，所有 BC 模型都是 0，不管。
- 已采的 self-play 数据（`/ytech_milm_intern/danglingwei/datas/RDJ_selfplay/`）：

| 录制目录 | 策略 / layout | 条数 | 成功 | 过程分 ≥ 50 |
|---|---|---:|---:|---:|
| `records_v1` | 34962 base，seed 0 layout 10–39，全部 54 配置 | 783 | 91 | 101 |
| `records_v2` | 34962 hide_track，seed 集 1，12 重点配置 | 310 | 114 | 76 |
| `records_v2b` | 34962 hide_track，seed 集 2，12 重点配置 | 253 | 67 | 44 |

  录制格式与转换脚本：`metiswam4d/eval/rdj_deploy.py`（Recorder）、`scripts/data_prep/robodojo/rdj_rollout_dataset.py`（`build` 用 SAPIEN 渲染机器人深度 / 掩码 + FK Track，`finalize` 出 `index.jsonl`，`status` 看计数；smoke 已验证 `RDJEpisodeDataset` 能直接读）。训练配置底稿 `configs/stage3_robodojo_v4_selfplay.yaml`（从 anneal 最新 ckpt 仅权重初始化、常数 LR、30 min 一 ckpt 保留 12 个、`data.robodojo.root` 指向 `RDJ_selfplay/dataset_v1`——需要改）。

## 二、目标

- **基础要求（必达）**：MetisWAM4D 在本地 420 条协议上的 Overall SR 明确超过 OpenWAM-α——本地复现 9.58 / 官方榜 **11.92**（Score 15.40 / 17.18），且不是噪声级的超过（10 条/任务协议下 Overall 的标准误约 1–1.5 分，所以至少要到 **13+**）。
- **冲刺目标**：逼近或超过 RoboDojo 官方榜单上最新的 WAM SOTA。榜单（2026-10-01）：PhysicalRSI **31.38**、Simate-beta 27.96、VPP2-Preview 25.62、Liber-0 Preview 25.52、Liber-0 Lite 24.23、GPT-6-Astra 22.48、OpenWAM-α 11.92。榜单口径是官方全量评测，与本地 10 条/任务协议只能近似对比，但方向一致：**冲刺线是 Overall SR ≥ 22（超过榜单最后一名 SOTA 级模型），理想是 ≥ 28–31**。这意味着要把 SR 翻一倍以上，一轮 self-play 做不到，要多轮迭代并叠加物体 Track。
- 当前起点：anneal 34962 + hide_track = 10.75 / 17.14，与 OpenWAM-α 官方差 1.2 分，与 SOTA 差 20 分。

## 三、方法：分治，降低每轮优化成本

全量优化一轮 = 35 个任务的数据训练 + 420 条评测（约 4 h），且 54 个配置里多数是 0/10、没有自举信号的任务，算力花在上面是浪费。因此**只在单独抽选出的 9 个任务 / 12 个配置上做训练—评测—自采的闭环迭代**：

- **训练**只用这 9 个任务的官方示教 + 它们的自采数据（数据量约为全量的 1/4，同样步数下每个任务见到的样本是全量训练的 4 倍，收敛快）。
- **评测与比较只在这 12 个配置上做**：协议 layout 0–9、与 `simeval_34962_hide` 同 layout 同 seed 配对，每轮 120 条（base / hide_track 各一遍 240 条，约 1.5–2 h），用"12 配置成功数 vs 旧权重的 24/120"作为每一轮是否有效的唯一判据。
- **其余 33 个任务不训不评**，直接沿用 `simeval_34962_hide` 的逐任务结果；最终 Overall 按任务级路由拼出来（每个任务取 旧权重 / 新权重 × base / hide_track 中更好的那个；评测时任务名已知，这是合法的部署方式），得到 420 条口径的五维度与 Overall SR / Score，与 OpenWAM-α、Janus 40k、榜单对照。
- 新权重在这 12 个配置之外的任务上可能退化，无所谓——那些任务走旧权重。

重点配置：`match_and_pick_from_conveyor`、`cover_blocks`（Memory，每条成功值 0.33 分）；`build_tower`、`pour_balls_into_vase`、`insert_tubes`（Precision，0.25）；`play_tic_tac_toe`（Long-Horizon，0.25，只有 hide_track 能做）；`stack_blocks`、`pour_liquid_into_cup`、`fold_clothes` 各含 `_random` 配置（Generalization，0.17）。当前这 12 配置在协议上 24/120。`stack_blocks_random` 自采 0/49 无法自举，进训练集也无妨但不期待提升。

三条路线按优先级：(1) 闭环自采回灌（filtered BC / expert iteration）；(2) 物体 Track 通道；(3) 控制精度 / 分辨率（最后、可不做）。**不做 agentic / VLM 任务分解。**

## 四、执行步骤（串行，t2 只有这一台；训练 8 卡满占时不能跑仿真）

1. **构建数据集 `RDJ_selfplay/dataset_v2`**（~0.5 h，任一 GPU，SAPIEN 需 `VK_ICD_FILENAMES` 指向 sapien 自带 `nvidia_icd.json`）：对 `records_v1`（只取 9 个重点任务）、`records_v2`、`records_v2b` 分别 `build --tag v2`（接受成功或过程分 ≥ 50），`finalize --tag v2 --repeat 3`。然后把 `index.jsonl` 的官方 train 行筛到 9 个重点任务（val 行保留原样供 `val/*`），记录每任务示教 / 自采条数。
2. **训练 round 1**（~4–5 h）：复制 `stage3_robodojo_v4_selfplay.yaml` 为 `stage3_robodojo_v5_focus.yaml`：`data.robodojo.root` 与 `robodojo_encoder.depth_stats` 指到 `dataset_v2`，`init.initialize_from` 指到 `checkpoints/step_0037951`，`output_dir` 为 `.../stage3_robodojo_v5_focus/`，`max_steps` 5000，其余不动。启动按第六节"GPU 保活"的规则：停保活（精确命令行 pkill）与起训练放在同一条命令里，训练退出后在同一 tmux 命令串末尾自动拉回保活（`scripts/robodojo/launch_t2.sh` 是从开发机 ssh 到 t2 用的，你在本机就把它 heredoc 里那段逻辑本地执行、tmux 会话名 `rdj_train`）。核对：首条日志 `time/data`≈0、`loss/action` 在 0.003–0.01、显存 < 80 GB；第一个 ckpt 看 `vis/step_*/` 与 `complete.json`；之后每小时看 `val/action` 与 `train_log.jsonl`。
3. **评测 round 1**（~2 h）：`scripts/eval/export_bf16.py --checkpoint <最后 3 个 ckpt 逗号分隔>` 得均值 bf16 权重；`rdj_campaign prepare --only <12 配置> --policy-variants '{"base": {}, "hide": {"hide_track": true}}' --episodes-per-round 2 --clients-per-gpu 3`（`env_seed 0`、`layout_offset 0`、不录制 → 协议 layout 0–9），`run` 8 卡 24 路；`summary_variants` 出配对表。用 `scripts/eval/rdj_paired_report.py` 把新结果与 `simeval_34962_hide` 逐任务对照，并拼出任务级路由后的 420 条 Overall SR / Score 表（含五维度）。
4. **决策**：若 12 配置成功数 ≥ 24 + 6（即至少 +6 条），进入 round 2；若 +0 ～ +5，仍进入 round 2 但把自采 `--repeat` 提到 5、`max_steps` 提到 8000；若为负，停止自采路线，转步骤 6 的物体 Track 并写明诊断。
5. **round 2**（采 ~4 h + 训 ~4 h + 评 ~2 h）：用 round 1 权重和它更好的推理方式续采（`rdj_campaign prepare --record ... --env-seed 1/2`，续跑现有 `selfplay_collect_v2 / v2b` 的剩余 layout，再加 `--env-seed 0 --layout-offset 40`），录制模式每卡 ≤ 4 个 Isaac 客户端 + 2 个策略服务（显存 9–11 GB/客户端）；新成功集并入 `dataset_v3`，从 round 1 最新 ckpt 续训，再评。预期每轮 +3–6 分；连续一轮不涨就停。
6. **物体 Track（在 GPU 被训练占用的时段用 CPU 做数据）**：`/m2v_intern_v3/danglingwei/ytech_face_algo_ssd_danglingwei/datas/GeoRoboDojo_JanusAct4D_Imperfect/_pipeline/work/<task>/episodeN/` 已有 4 相位 stride-4 RAFT 光流（`flow/phaseP.h5: forward_flow_px [n/4,240,320,2]`、`source_frame_index`）、DA3 深度（`da3.h5: depth_m [n,240,320] f16、intrinsics、extrinsics_w2c`），`<task>/train_4d/masks/episodeN.h5` 为机器人 + 被操作物体双通道位掩码（`head_camera/mask_bits [n,2,240,40]`，little bitorder；`cover_blocks` / `stack_blocks` 物体通道近乎为空，物体改按 |flow| 阈值判定）。为 9 个重点任务写 `track4d.h5` v2：机器人像素沿用 `RDJ_MetisWAM4D` 的 FK Δuvd 与 role 1；非机器人像素 Δu Δv = RAFT，Δd = DA3 在 (t+4, u+Δu, v+Δv) 与 (t, u, v) 的深度差，role 2 = 运动物体、0 = 静止背景。**输入侧（锚帧、head_mask、深度条件）保持机器人-only**，推理不新增 gap。训练改 `track_regions: {body: 1, object: 1, background: 0.2}`、`unknown_role: false`。作为 round 2 或 3 的叠加项单独训、单独评 base vs hide_track，判断 Action 是否开始从 Track 受益。先核对 DA3 深度与 SAPIEN 机器人深度在机器人像素上的尺度偏差并写进文档。
7. 精度 / 分辨率：只有 Precision 三任务两轮都不动时再考虑，且 24 小时内大概率不做。

## 五、里程碑（把第二节的目标换算到 12 个重点配置上）

12 配置每多 1 条成功对 Overall 的贡献：Memory 0.33 分、Precision / LH 0.25、Gen 0.17；起点 24/120、Overall 10.75。

| 里程碑 | 12 配置成功数 | 路由后 Overall SR（约） | 含义 |
|---|---:|---:|---|
| round 1 有效 | ≥ 30 | ≥ 12.2 | 超过 OpenWAM-α 官方 11.92 |
| 基础要求达成 | ≥ 36 | ≥ 13.5 | 超过 Alpha 且超出噪声 |
| 期望 | ≥ 48 | ≥ 16 | 重点任务成功数翻倍 |
| 冲刺 | ≥ 72（接近 12 配置全部 6/10 以上）+ 物体 Track 带来非重点任务的提升 | ≥ 22 | 进入榜单 SOTA 区间 |

每轮结束都按这张表报告处于哪一档。

## 六、硬性约束

- Python：训练 / 评测服务 / 数据脚本用 `/usr/bin/python3.10`；Isaac 客户端用 RoboDojo 的 python3.11 运行时；离网，不能 pip。
- **GPU 保活（最重要的运维约束）**：集群按 GPU 利用率回收机器，空闲几十分钟就可能被收走，所以 t2 上任何时刻都必须有"真实负载"或保活进程 `wangrunqi_nvml_busy.py`（每卡约 7 GB 显存、98% 利用率的 CUDA 睡眠核）。规则与命令：
  - 现状：PID 21 `python /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py --gpus all --size 2000 --pid-file /m2v_intern_v3/danglingwei/logs/high16_bdy.pid --log-file .../high16_bdy.log`，是 pod 启动时拉起的；它的 pid-file 文件实际不存在，所以停它只能按**精确命令行**：`pkill -f '^(/usr/bin/)?python3? /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py --gpus all'`（`scripts/robodojo/launch_t2.sh` 里就是这条），禁止 `pkill python` / `killall`。
  - **什么时候停**：只在 8 卡训练 torchrun 启动前的那一刻停（它占 7 GB/卡，训练要 78 GB），停完 3 秒内就起训练，不要让 GPU 空着。数据构建、权重导出、仿真评测 / 自采期间**不停**——它们只用部分显存，保活负责填满利用率缺口（campaign manager 还会通过 `--keepalive-pid-file` 守护：保活死了按原命令拉起）。
  - **什么时候拉回**：训练进程一退出（正常结束或崩溃）立刻拉回，写法是把拉起命令接在同一个 tmux 命令串末尾（`launch_t2.sh` 的做法）：`setsid nohup /usr/bin/python3 /m2v_intern_v3/danglingwei/wangrunqi_nvml_busy.py --gpus all --size 2000 --pid-file /m2v_intern_v3/danglingwei/logs/t2_after_rdj_<时间戳>.pid --log-file /m2v_intern_v3/danglingwei/logs/t2_after_rdj_<时间戳>.log > /dev/null 2>&1 < /dev/null &`，这样崩溃也会自动恢复。拉回后 `nvidia-smi` 核对 8 卡各约 7 GB、利用率 90%+。
  - 同一时间只能有一个保活实例（两个会多占 7 GB/卡）；起之前先 `pgrep -f wangrunqi_nvml_busy` 看有没有。
  - 任何长任务（训练、campaign）停下后，第一件事是确认保活在；每次切换阶段（训练→评测、评测→训练）都要在同一条命令里完成"停旧负载 + 起新负载"，中间不留空窗。
- 不碰 t4 以及任何其他机器；t2 上没有别人的作业。不递归扫描 Ceph 大目录（`getfattr -n ceph.dir.rfiles` 预检）。输出全部放 `/ytech_milm_intern/danglingwei/outputs/MetisWAM4D_260921/<run>/` 或 `/ytech_milm_intern/danglingwei/datas/RDJ_selfplay/`，`/m2v_intern_v3` 只放代码与文档。
- 并行 campaign 用不同 `RDJ_BASE_PORT`（33380 / 33480 / 33680 / 33780 已用过）；停 campaign 给 manager PID 发 TERM，**不要直接 kill tmux**（manager 收 SIGHUP 会留下孤儿客户端，需按 `^/usr/local/robodojo-python-runtimes/.*rdj_client` 与 `^/usr/bin/python3.10 -m metiswam4d.eval.rdj_policy` 精确清理）；禁止宽泛 `pkill python`。
- 长任务一律进 tmux；每一步启动后不算完成，要核对首条日志 / 首个产物。
- 单个坏样本不能杀掉作业；仿真客户端崩溃由 manager 重试，连续失败才升级。

## 七、记录与汇报

- 所有决策、配置、结果追加写进 `docs/2026-09-30-RoboDojo闭环仿真评测-实现与运行记录.md`（中文，先结论后细节，数字与路径给全，不写过程噪声）；每完成一个阶段更新文末"交接"小节（目标、当前代码 / 配置、在跑任务及日志位置、已验证结果、下一步与阻塞项），我回来后只看这一节。
- 最终给我一张表：每个重点配置的 旧 / 新(base) / 新(hide) 成功数，路由后的五维度与 Overall SR / Score，与 Alpha 本地 / 官方、Janus 40k 对照；以及"什么有效、什么无效、为什么"的三句话结论。
- 不确定的原因就说不确定，不编造；不利的数字直接给。

现在开始：先用不超过一屏的篇幅列出你的执行时间线（含预计耗时、每步的核对点与失败时的退路），写进 docs 后立刻执行步骤 1，不用等我。
