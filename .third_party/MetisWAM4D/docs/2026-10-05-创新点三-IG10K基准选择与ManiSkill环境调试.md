# 创新点三（视频意图与本体解耦）：IG-10K 基准选择与 ManiSkill 仿真环境调试（2026-10-05）

## 1. 基准选择

创新点三需要的故事是：某些任务上不读（或耦合地读）人类示范时成功率很低，用解耦的人类示范后成功率提升。RoboTwin 标准榜我们已到 93.2%，加示范说明不了问题。调研 Zero-WAM 及同类 in-context / 人类视频条件工作的数据后，主基准定为 **IG-10K 仿真 Imitator Arena**。

| 依据 | 内容 |
|---|---|
| 现有方法低分 | 论文 Table 3：15 个基线（π0.5 / GR00T / RDT / OpenVLA / XSkill / UniSkill / ACT / DP / VQ-BeT）在 5 个未见任务上零样本 SR 最高 0.13，各级别平均约 0.06；见过任务最高 0.81；10 条示范微调最高 0.85 |
| 级别 = 解耦程度 | L0 可抄轨迹；L1 布局变化；仿真 L2 换执行手臂（人手与机器人本体不对应）；L3 功能替代 |
| 示范确实被读 | Table 10：ACT/DINOv2 见过任务正确示范 0.81，无关示范 0.30 |
| 论文结论 | 呼吁"把示范作为目的证据保留、能跨物体替换存活的中间表示"的接口 |
| 数据 | 本地 IG-10K：真人示范 3256 + 800 条（手部 / 物体 Track4D 已做）、真机 11,220 集（Track4D + 动作已做）、仿真 10K 集（200 目录，未处理）；`examples/baselines/lerobot_dataset/task_mapping.json` 给出 human ↔ robot ↔ sim 任务对应 |

评测协议：5 个见过任务（StirSpoon、FoldTowel、PlaceMugRack、PlaceFileFolder、PlacePlateRack）+ 5 个未见任务（PickRemoteControl、ScanMilkBox、PourKettle、PickFood、FoldBox）× L0–L3，仿真每组 10 次，自动指标 SR / Sub-SR；设置分零样本、从头 10 条、预训练 + 10 条微调。

其他数据的结论：

| 数据 | 结论 |
|---|---|
| HumanGen-RoboTwin（Zero-WAM） | 2499 对 / 994 条 kling-v3 生成人类视频，5 s、12 fps、320×448，Wan2.2 VAE latent（与本项目同 VAE），43/7 任务划分；可作 RoboTwin 跨任务协议的备选，未选为主线 |
| HumanGen External | 41K 对，只有真机机器人数据；Zero-WAM 另有 30K 内部数据未公开 |
| H&R、RH20T、Vid2Robot | 只有真机，平台不匹配 |
| HOST、WAM-TTT | 只开源代码或数据未公开；HOST（共享任务进度流形 + DTW/TCC 对齐）是进度对齐的最近邻方法 |
| WatchAct、AGNOSTOS、H2RBench、BPP | LIBERO 仅评测 / RLBench 无人类视频 / Isaac Lab 4 个单任务 / 示范为机器人 |

## 2. ManiSkill 仿真环境（dev 机已跑通）

| 项 | 内容 |
|---|---|
| 代码 | `/m2v_intern_v3/danglingwei/codes/TheImitatorGame_261005`（commit `d6d16ec`，2026-09-16；仓库自带 ManiSkill 3.0.0b20 fork） |
| Python | `/usr/bin/python3.10`（sapien 3.0.0b1、torch 2.7.1、gymnasium 0.29.1 与仓库要求一致）；仓库 `pyproject` 要求 3.12 + uv，只在跑其官方基线时需要 |
| 缺失依赖 | 装在 overlay `/ytech_milm_intern/danglingwei/envs/ig_py310_overlay`（`pytorch_kinematics_ms`、`arm_pytorch_utilities`、`pytorch-seed`、`fast_kinematics==0.2.2`、`dacite`、`tabulate`、`dm-tree`），不改共享 site-packages；系统 mplib 为 0.2.1（仓库指定 0.1.1，只影响运动规划采集） |
| 环境变量 | `scripts/ig10k/ms_env.sh`（PYTHONPATH、MS_ASSET_DIR、VK_ICD_FILENAMES 指向 sapien 自带 `nvidia_icd.json`） |
| 资产 | 压缩包在 `/ytech_milm_intern/danglingwei/datas/IG-10K-Assets/`（`assets` / `sketchfab` / `partnet_mobility`，大小与 HF 一致）；解压到运行机本地盘 `~/.maniskill/data`（dev 已解压，partnet 12 GB）。`robotwin/{objects,background_texture}` 软链到本地 `/m2v_intern_v3/danglingwei/files/RoboTwin_260625/assets/`：IG 代码引用的 53 个 RoboTwin 物体全部存在，渲染与数据集首帧一致，14.8 GB 的 `robotwin.tar.zst` 不下载。新机器只需解压三个包并建同样的软链 |
| 级别切换 | 环境变量 `MANI_SKILL_L1/L2/L3` + `L0_L3_utils.set_l{1,2,3}_enabled`；L3 为独立注册名 `TwoRobot*L3-v1` |
| L2 镜像 | L2 自动对场景物体做左右镜像；必须 `L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)`（机器人基座不换位），与官方评测和数据集动作一致。库默认值为 True，会使两台机器人互换，数据集回放失败（PickFood L2：True 时 reach 峰值 0.21 / 0.29、失败；False 时成功；True + 交换两臂动作也成功） |
| 接口 | 双 Panda（`panda_wristcam` × 2），动作 Dict，每臂 `pd_joint_pos` 8 维（7 关节 + 夹爪，夹爪 1 = 张开）；数据集 16 维动作 = [臂 0 的 8 维, 臂 1 的 8 维]，18 维状态 = 每臂 7 关节 + 2 指；观测 `rgbd`，相机 zed2i 224×224 + cam1–3 + 腕部；info 含 `success` 与分阶段奖励（`R1_reach…`，Sub-SR 的来源） |
| 冒烟 | `scripts/ig10k/ms_smoke.py`：评测用的 10 个任务 × L0–L3 共 40 组全部可 make / reset / step，无缺资产，CPU 后端约 0.05 s/步；拼图 `outputs/MetisWAM4D_260921/ig10k_sim_smoke/eval10_grid.png`；与数据集 L0 首帧相机位姿、分辨率、资产外观一致（`TwoRobotPickAppleBasket-v1/dataset_vs_sim_L0.png`） |
| 回放校验 | `scripts/ig10k/ms_replay_check.py`：数据集不存 seed，采集器从 0 递增、规划失败即跳过，按首帧与 reset 画面最小 MSE 找回 seed（例：PickAppleBasket L0 第 2 集 → seed 10）；按上述镜像设置开环回放 15/15 成功（PickAppleBasket L0 ×3、FoldBox L1 ×2、PourKettle L3 ×2、PickFood L0 ×1 / L2 ×1、PickAppleBasket / FoldBox / StirSpoon L2 各 ×2）。结果在 `outputs/MetisWAM4D_260921/ig10k_sim_smoke/replay/` |
| 部署 | 新机器 `bash scripts/ig10k/setup_assets.sh`（解压三个包到本地盘 + RoboTwin 软链，约 10 分钟）。t4、t5 已部署（2026-10-05 18:12），各自回放 2/2 成功 |
| 重渲 | 任务构造参数 `hi_res=True`：zed2i 1280×720（16:9，= 512×288 的 2.5 倍）+ cam1–3 640×480。同 seed 下 `hi_res` 开关不改变仿真状态（114 维 state 差 0），可用 224 图找回 seed、再高分辨率重放。`obs_mode="rgb+depth+segmentation"` 的分割 id 对应各机器人连杆与各物体 actor（`segmentation_id_map`），可直接给角色真值；4 相机高分辨率约 0.13 s/步（低分辨率 RGB-D 0.05 s/步，t4 实测） |
| 官方评测 | 清单 `examples/baselines/lerobot_dataset/eval/exp_list/{seen,unseen}_5tasks_env_list.txt`；官方 `eval_envs.reset()` 不给 seed，初态随机，对照实验需自行固定 seed |

网络：GitHub 走海外代理 `oversea-squid1.jp.txyun:11080`（约 5.9 MB/s，国内代理约 40 KB/s）；PyPI 两个代理都可；hf-mirror 走国内代理 `10.66.29.113:11080`（6–8.5 MB/s），d1 直连 hf-mirror 不通。

## 3. 仿真真值 4D 数据（`scripts/ig10k/sim_gt4d.py`）

决定（2026-10-05）：动作用原生 16 维关节位置（评测控制器 `pd_joint_pos`）；只渲 zed2i 主视角。

| 项 | 内容 |
|---|---|
| 流程 | 224 首帧找回 seed（从上一集 seed + 1 顺序扫描，MSE < 25 即回放验证，失败回退到窗口内排名前 3 及宽扫描）→ zed2i 1280×720 RGB + 深度 + 分割重放 → 只保留回放成功的集，失败记 `_logs/bad_replay.jsonl` |
| 视角 | 1280×720 的 zed2i 与数据集 224×224 同相机、同垂直视场角（约 99°），224 版是其中央 720×720 区域；训练与闭环评测统一用 16:9（512×288） |
| 产物 | 每集一个 h5：`rgb`（512×288 JPEG）、`depth_mm` / `part` / `role`（180×320 网格，1280×720 的 4×4 像素中心采样）、`delta_uvd`（t→t+4，du/dv 网格像素、dd 米）、`qpos18` / `actions16` / `sim_qpos18`、`entity_pose`、内外参；part 0 静止 / 1 臂 / 2 夹爪（hand / finger / tcp / camera 连杆）/ 3 运动物体（本集位移 > 2 mm 或转角 > 1° 的 actor 或关节连杆），schema `metiswam4d.ig10k_sim_track4d.uvd.objects.v1` |
| 冒烟 | PickAppleBasket L0、FoldBox L2、FoldTowel L1 共 10 集全部成功；落点深度误差中位数 0.4–0.6 mm、96–98% < 5 mm；FoldBox 只有被折的纸板 `link_1` 判为运动物体，FoldTowel 为毛巾关节体三个连杆；开局物体落定的小位移会把篮中香蕉、柠檬判为运动物体（真实运动）。面板 `datas/IG10K/preprocessed/sim_smoke/qc_*.png`；约 25–40 MB / 集 |
| 渲染设备 | `shader_pack="rt-fast"` 配显式 `render_backend="cuda:N"`（N > 0）会在首次取图时卡死；用 `CUDA_VISIBLE_DEVICES` 选卡 + 默认 `gpu` |
| 首次运行 | 2026-10-05 19:03 起 t4、t5 tmux `ig_sim4d`，每台 48 worker（GPU 轮转，OMP 1）；worker 从各自偏移遍历全部 200 目录，`_locks/` mkdir 锁 + `_done/` 完成标记；进度 `scripts/ig10k/sim_gt4d_status.py`。头 2 分钟 245 集、0 失败；每卡显存 7 → 26 GB（`CUDA_VISIBLE_DEVICES` 分卡生效） |
| 输出 | `/ytech_milm_intern/danglingwei/datas/IG10K/preprocessed/sim/`，日志 `_logs/<目录>.jsonl`、`_logs/workers/` |
| 事故（19:25 / 19:30） | 两台 pod 先后因内存超限被杀并重启（13330 sshd 与保活进程一并消失）：t4 19:25:17 worker 报 `vk::Device::createFenceUnique: ErrorOutOfHostMemory` 后再无日志；t5 每集耗时 45 → 128 s，19:30:15 后无日志。pod 的真实额度看 cgroup（`memory.limit_in_bytes`、`cpu.cfs_quota_us`），`free` / `nproc` 显示的是宿主机：t4 / t5 为 800 GB / 80 CPU，dev 200 GB / 24 CPU，d1 100 GB / 16 CPU。48 worker 的 RSS 合计约 106 GB，远低于 800 GB，说明生产运行中存在 RSS 之外、随运行时间增长的内存占用（单进程在 dev 上连续 40 次 reset、跑 6 集，RSS 稳定在 2.2 GB，未复现）。根因未确认，待用 `scripts/ig10k/mem_probe.sh`（cgroup 分项 + worker RSS）在小规模多 worker 运行中定位。已落盘 2131 集（66 GB）完好；96 个陈旧锁与 2 个 `.tmp` 已清理；pod 重启后本地资产丢失，需重新解压 |
| 续跑配置 | `launch_sim_gt4d.sh` 按 cgroup CPU 配额取 2/3 为 worker 数，每 20 s 记工作集（usage − total_inactive_file），超过上限 `MEM_FRAC`（默认 70%）结束最新 worker；worker 收 SIGTERM 时释放目录锁（d1 上验证）；日志 `_logs/workers/<host>_monitor.log` |

冒烟 `sim_smoke2` 中 FoldTowel 目录被两个进程写过（第 1–3 集早于本轮第 0 集落盘），另一个写入进程未能确认是哪次运行；开跑前 `rm -rf` 输出目录会删掉其他进程持有的锁。同 seed 回放确定、h5 先写临时名再改名，内容未受影响。之后日志记 pid，启动前核对两机无残留 worker，运行期间不清理输出目录。

## 4. 下一步

1. 评测服务：按本项目策略服务协议写 IG 评测客户端（固定 seed 列表、人类示范按 `task_mapping.json` 取、记录 SR / Sub-SR），先用数据集动作回放当"oracle 策略"验证整条链路。
2. 仿真数据转本项目格式：zed2i RGB-D 224×224、18 维 qpos / 16 维关节动作；仿真 Track4D 用"找回 seed + 回放"在环境里取机器人 / 物体分割与位姿，得到真值几何。
3. 机器选择：dev 只做调试；批量回放与评测放空闲训练机（需在该机解压资产并建 RoboTwin 软链）。
