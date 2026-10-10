# Git、处理数据 Release 与恢复

## 三类资产

| 内容 | 保存位置 | 用途 |
|---|---|---|
| 脚本／配置／树状文档 | [真机训练分支](https://github.com/Yutenji-Nyamu/UR5e-RoboTwin-Real/tree/codex/geowam-real-training) | 解释与复现处理、初始化、训练 |
| 原始 RGB-D | [数据仓库](https://github.com/Yutenji-Nyamu/data_ur5e_gwam) 的 10 月 6／7 日 Release | 重跑视觉处理 |
| 已处理数据 | [RGB-D72 v1 Release](https://github.com/Yutenji-Nyamu/data_ur5e_gwam/releases/tag/processed-rgbd72-v1-20261010) | 直接读取训练缓存、复查几何 |

## 新 Release 内容

五个按任务分开的 `train-*.tar.gz` 加一个 `shared.tar.gz`：2,405 个活跃训练窗口、清单、归一化、文本缓存、各 episode 元数据与数据契约。合计约 1.08 GB。

`geometry-qa.tar.gz`：最终全分辨率 mask／role、处理后的 UVD／有效性、逐段质量统计及 mask／flow／VAE 预览，约 0.22 GB。

`processed_downloads.json` 和 `SHA256SUMS.txt`：文件大小、SHA-256、源清单哈希与来源版本。数据仓库的 `processed/rgbd72-v1/README.md` 提供恢复入口。

## 哪些处理结果需要保留

已有模型格式的继续训练使用 train＋shared；分析分割、有效性或重做编码时再取 geometry-qa；更换 SAM／RAFT 时使用原始 RGB-D、处理脚本及配置。服务器保留旧缓存和抽样原始 flow 对照，Release 选择当前训练清单和最终处理结果。

模型权重、训练优化器检查点和下载缓存继续放服务器模型／运行目录。发布脚本：[export_processed_release.py](../../../scripts/geowam/export_processed_release.py)。本次云端完成信息见 [交付记录](../16_云端交付.md)。
