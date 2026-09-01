# 工作站存储

[English](../STORAGE.md)

盘点日期：2026-09-01。未删除任何历史项目或模型数据。

## 当前布局

| 设备 | 文件系统 | 状态 | 定位 |
|---|---|---|---|
| 1TB Lenovo NVMe | ext4 | 可用937 GiB，已用839 GiB，剩余52 GiB | 系统、源码、Git、Conda/venv、活跃小文件 |
| 4TB Seagate HDD | NTFS3 | 挂载到 `/data`，已用299 GiB，剩余3.4 TiB | 数据集、录制、checkpoint、归档 |

HDD 已按 UUID 写入 `/etc/fstab`，使用 `nofail` 和 systemd automount。
`zhangw`、`ur5`、`wlf` 均加入 `robotdata`；`/data/robotics` 为组可写并继承组。
修改前的 fstab 备份位于 `/etc/fstab.codex-backup-20260901`。

保留 NTFS 是为了不破坏盘上已有跨平台数据；源码、Conda、venv 以及高度依赖
Unix链接/权限的工作负载继续留在ext4。

## 根盘占用归属

| 账户/区域 | 约占用 | 主要内容 |
|---|---:|---|
| `zhangw` | 480 GiB | `ScutPythonProject` 354.7 GiB，主要为一个RoboTwin策略树 |
| `ur5` | 272.4 GiB | RoboTwin/Pi0.5工作区130.2 GiB、缓存90.5 GiB、Anaconda45.3 GiB |
| `wlf` | 38.2 GiB | 缓存31.9 GiB，主要是pip和Hugging Face |

最适合迁移的常规数据目录：

1. 旧 DETwinVLA checkpoints：约208.1 GiB；
2. DexVLA checkpoints：约89.5 GiB；
3. Pi0.5 checkpoints与训练数据：约59.3 + 15.5 GiB；
4. 历史 RoboTwin ACT/processed data：数十GiB。
5. 下载目录顶层的三个压缩包/模型分片：合计约17.4 GiB。

它们可迁至 `/data/robotics/shared`，复制并校验后再用软链接接回。当前GPU和进程
检查没有训练任务，但迁移仍应作为独立I/O任务执行，不与真机调试同时进行。

两个最大的checkpoint目录内部都没有软链接，是最高收益且最安全的候选；迁移
两者可为NVMe释放约297.6 GiB。

两个账户的pip缓存合计约30.5 GiB，Conda包缓存也可重新下载，通常应清理而不是
归档。Hugging Face缓存、完整环境和Git工作树不盲目迁到NTFS。

## 数据盘目录

```text
/data/robotics/
├── ur5e-real/{raw,converted,checkpoints,logs}
├── shared/{datasets,models,checkpoints,archives}
└── staging
```

本机 `configs/lab.yaml` 已将 `collection.data_root` 设置为
`/data/robotics/ur5e-real`。数据命名、保留和校验迁移规则见
[`DATA_MANAGEMENT.md`](DATA_MANAGEMENT.md)。
