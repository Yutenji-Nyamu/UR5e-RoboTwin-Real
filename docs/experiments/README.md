# 实验资料归档 / Experiment evidence

2026-09-12 起，小型实验资料尽量完整进入 Git，不再只提交手写摘要。
当前快照包含成功、失败和调试记录；不因结果不好而省略。

## 已保存什么

| 归档位置 | 来源与内容 |
| --- | --- |
| `evidence/repository/logs/` | π0.5训练/评估/资源监控、batch短测、推理记录、RLT检查 |
| `evidence/repository/outputs/` | 所有现有Loss图、统计JSON、数据复核图，包括历史版本 |
| `evidence/repository/checkpoints/` | recipe、norm、contract、验收、参数布局等小型元信息 |
| `evidence/repository/configs/lab.yaml` | 当前工位实际配置原文快照；日常配置入口仍是仓库根目录的 `configs/lab.yaml` |
| `evidence/data_root/` | DATA_LOG、全部session结果、动作/夹爪/同步CSV、DP日志/图、数据集元信息和小型诊断数组 |

[MANIFEST.json](evidence/MANIFEST.json)列出每份文件的原始位置、采样时间、源/归档SHA-256和字节数。
这是逐文件快照，不是运行的原子快照：正在训练的插座实验仍会继续产生日志和checkpoint。
JSONL若恰好读到未写完的末条记录，只归档完整前缀，并明确记录省略字节数；源文件不变。
历史绝对路径和设备参数保留原貌，不代表另一工位可直接使用。

## 后续一条命令更新

在项目根目录，用项目Python环境执行：

```bash
python scripts/archive_experiment_evidence.py
```

数据盘不同可加 `--data-root /实际数据根目录`。每次实验收尾及提交实验记录前运行一次，
检查 `git diff` 后连同代码/文档提交推送。没有变化的文件不重写；旧快照不会自动删除。
原 `logs/outputs/checkpoints` 仍是运行目录，归档副本独立版本管理，正在训练不会不断弄脏 Git。

单文件上限10 MiB。成套原始相机图像、视频、HDF5/Parquet/Zarr数组载荷、模型权重及optimizer
分片不入Git；图像虽单张小，整批已达GiB级。实验复核图和小型NPZ诊断不在此排除范围。
环境/缓存仍整体忽略；RoboTwin通过已提交的lock、bootstrap和补丁复现，不重复提交第三方工作树。
归档不是完整数据/模型备份，不能凭这些元信息恢复权重。实际密钥文件不在归档范围内。

存储快照与后续增长来源见[工作站存储](../zh-CN/STORAGE.md)。

## English summary

Small experiment evidence is versioned here, including unsuccessful runs, raw numeric CSVs,
dataset manifests, loss/inspection images, diagnostics, checkpoint metadata and the workstation config.
Run `python scripts/archive_experiment_evidence.py` before committing experiment results.
Original files are never edited. The manifest records source paths, timestamps, byte counts and SHA-256.
Live runs are per-file snapshots, not completed-run claims. Incomplete JSONL tails are explicitly accounted for.
The exporter caps files at 10 MiB and omits bulk image datasets, model/data payloads, environments and caches.
Native model storage fragments are not a usable checkpoint backup. Upstream code is reproduced from its lock and patches.
