# GeoWAM 真机：2 机实施入口

工作目录：`/data/chenyiteng/projects/geowam-real`。代码分支：`codex/geowam-real-training`。范围为 2 机物理 GPU 6/7 上的数据处理、训练准备和离线推理；机器人执行后续迁移到现场机。

## 已完成
- UR5e、数据、稿件和 Gitee 模型仓库均已检出；Gitee 已配置专用 SSH 密钥和官方主机指纹，可持续 fetch/pull。
- 最新核对的 Metis 提交为 `1392ab585874701485c32372828a873bd0ac7644`，2026-10-09；完整来源见 `sources.lock.json`。
- 数据元数据逐条审阅：87 条成功示范，80 条 RGB-D，7 条 RGB；五任务数量为 23/16/17/16/15。原始 PNG 从 12 个 release 包恢复，进度见 workspace 的 `runs/data-download-status.json`。
- SAM 3.1 与 RAFT 已下载并完成首段真实片段试跑；细节见 `05_首段真机预处理验收.md`。SAM 3.1 魔搭镜像已核对：其 SHA256 对应的 LFS pointer Git OID 与 Meta 官方一致。RAFT 使用 torchvision 官方 Large 权重。
- Alpha 初始化形状核对完成：Video 825 个张量、Action 824 个、本体输入 2 个均匹配。Track 另做宽度/层数转换；新增模块按源码初始化。实际权重加载在下载完成后进行。
- RLT 的 GPU6/7 owner、driver 和 Ray actors 已交接退出，checkpoint 保留；原回退入口已停用。每次 GPU 启动前刷新占用与身份。

## 接下来按顺序
1. 首段 SAM 3.1、RAFT、UVD 可视化已跑通；继续扩展到整段及另外四任务。
2. 完成每任务 1 条审阅，再按已确定的 72 条 RGB-D 训练列表批量缓存。
3. 接入完整 Video/Track/Action 的 H32 样本、载入 Alpha、重算 UR5e 归一化，做短训练与离线动作接口测试。
4. 按共同训练 episode 列表开展五任务训练，保存迁移到真机的权重、动作接口和预处理配置。

## 文件入口
- `01_数据处理规划.md`：原始数据、缺深度、分割调试、UVD。
- `02_初始化与训练规划.md`：每个模块的初始化来源和训练顺序。
- `03_重启交接.md`：服务器、Git、GPU、下载和环境的续接方式。
- `04_来源与待落实项.md`：代码证据、公开原始来源与接口决策。

大文件置于 workspace 的 `data/models/cache/runs/envs`；代码、配置、规划在本分支追踪。分支当前在 2 机本地保存，包含项目内部规划。Overleaf 本轮新增、修改、删除均为 0。

## 2026-10-09 用户确定的分组

换桌布 5 条、彩色光照 5 条归为 OOD，已从训练列表排除；原始数据保留。训练列表确定为 72 条实测 RGB-D；另 5 条常规场景 RGB-only 已排除。实际分组由 `configs/geowam/episode_policy.json` 固定，`build_episode_manifest.py` 输出 workspace 的 `data/manifests/`。完整几何流程读取 `train_rgbd.jsonl`，`train_rgb_only.jsonl` 保持空表；OOD 分组独立保存。以上更新优先于前文的待分类状态。

下载更新：SAM3.1、RAFT、Wan VAE、UMT5 与 tokenizer 已下载并校验。Alpha 早期 HF 连接超时，已保留约 5.27 GB 并启用独立续传，见 `runs/alpha-resume-status.json`；原始 PNG 包继续后台恢复。
