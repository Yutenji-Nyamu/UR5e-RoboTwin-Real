# 工作站存储

[English](../STORAGE.md)

盘点更新：2026-09-04。

## 当前布局

| 设备 | 文件系统 | 状态 | 定位 |
|---|---|---|---|
| 1TB Lenovo NVMe | ext4 | 可用937 GiB，已用429 GiB，剩余461 GiB | 系统、源码、Git、Conda/venv、活跃小文件 |
| 4TB Seagate HDD | NTFS3 | 挂载到 `/data`，已用712 GiB，剩余3.0 TiB | 数据集、录制、checkpoint、归档 |

HDD 已按 UUID 写入 `/etc/fstab`，使用 `nofail` 和 systemd automount。2026-09-04
已用 `ntfsfix` 修复MFT镜像并清除dirty标记，自动挂载重新实测通过。
`zhangw`、`ur5`、`wlf` 均加入 `robotdata`；`/data/robotics` 为组可写并继承组。
修改前的 fstab 备份位于 `/etc/fstab.codex-backup-20260901`。

保留 NTFS 是为了不破坏盘上已有跨平台数据；源码、Conda、venv 以及高度依赖
Unix链接/权限的工作负载继续留在ext4。

## 已迁移的大块数据

DETwinVLA（208.1 GiB）、DexVLA（89.5 GiB）、Pi0.5 checkpoint与训练数据
（约74.8 GiB）、ACT数据与模型（约21 GiB），以及下载目录的大型归档/模型分片
（约17.4 GiB）已经迁入 `/data/robotics/shared`。旧位置保留软链接，因此原命令
无需改路径；NVMe现有约461 GiB余量。

可重新下载的pip/Conda缓存适合直接清理；Conda环境、Git工作树和频繁产生大量
小文件的缓存不迁到NTFS。

## 速度边界

4TB盘是机械硬盘，顺序写入录制数据、HDF5、视频以及长期保存checkpoint很合适，
但随机读取大量小PNG会明显慢于NVMe。模型首次加载也更慢，加载进内存/GPU后影响
通常很小。默认把raw和checkpoint放在 `/data`；训练前转成HDF5或分片格式。只有
实测训练被I/O卡住时，才把当次活跃数据临时放到NVMe，结束后移回数据盘。

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
