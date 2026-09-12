# 工作站存储

[English](../STORAGE.md)

盘点更新：2026-09-12，15:28 CST附近的运行中快照；不含清理或迁移操作。

## 当前布局

| 设备 | 文件系统 | 状态 | 定位 |
|---|---|---|---|
| 1TB Lenovo NVMe | ext4 | 总量937 GiB，已用531 GiB（60%），剩余359 GiB | 系统、源码、Git、Conda/venv、当前π0.5模型 |
| 4TB Seagate HDD | NTFS3 | `/data`正常rw挂载，已用736 GiB（20%），剩余约3.0 TiB | 数据集、录制、DP checkpoint、归档 |

## 系统盘占用和增长

| 可读取部分 | 约占用 | 组成 |
| --- | ---: | --- |
| 当前项目 | 100 GiB | π0.5 checkpoint约70 GiB、环境/缓存约30 GiB；源码与实验小日志不是大头 |
| 当前用户Conda | 56 GiB | 环境48 GiB，pkgs缓存3.7 GiB，其余为base与工具 |
| `ScutPythonProject` | 37 GiB | RoboTwin约25 GiB、DexVLA约5.3 GiB、LLMs约4.4 GiB |
| 旧采集项目 `UR5e_DataCollection` | 33 GiB | RoboTwin约24 GiB、相机数据4.9 GiB、旧Git历史3.5 GiB |
| `/usr`、`/var`、`/opt` | 22、16、3.5 GiB | 系统和软件；`/var`含两份CUDA安装源共6.3 GiB、日志约1.9 GiB |

旧账户目录和部分系统目录未获读取权限，表格不是闭合的全盘总账，不把不可读目录算作0。
`du`对子目录有硬链接去重，环境/缓存子项不能机械相加成可释放空间。

当前主要长期增长源是checkpoint：单份约8.8 GiB。三个已完成π0.5实验各留500/1000两份，
合计约53 GiB；batch=2的3步短测还占8.8 GiB；本次插座已保存500步约8.8 GiB，
还计划保存1000/1500两份，预计再增加17.6 GiB。即使完成，系统盘仍约有341 GiB余量。
当前`.venv`还含基础模型约12 GiB、uv缓存约13 GiB、dummy测试模型3.1 GiB、HF数据缓存2.1 GiB。

因此不是迫近的满盘故障，但长期优先把不活跃checkpoint归档到`/data`；先考虑短测/dummy产物，
正式实验保留有用模型。迁移或清理需独立执行并验证路径，不在训练中移动当前模型/环境。
本次只做盘点和小型资料Git归档，未删除或迁移任何数据。

HDD 已按 UUID 写入 `/etc/fstab`，使用 `nofail` 和 systemd automount。2026-09-06
已用仓库内的 `ur5e-storage-repair` 修复MFT镜像、清除dirty标记并实测重新以 `rw` 挂载。
`zhangw`、`ur5`、`wlf` 均加入 `robotdata`；`/data/robotics` 为组可写并继承组。
修改前的 fstab 备份位于 `/etc/fstab.codex-backup-20260901`。

保留 NTFS 是为了不破坏盘上已有跨平台数据；源码、Conda、venv 以及高度依赖
Unix链接/权限的工作负载继续留在ext4。

## 已迁移的大块数据

DETwinVLA（208.1 GiB）、DexVLA（89.5 GiB）、Pi0.5 checkpoint与训练数据
（约74.8 GiB）、ACT数据与模型（约21 GiB），以及下载目录的大型归档/模型分片
（约17.4 GiB）已经迁入 `/data/robotics/shared`。旧位置保留软链接，因此原命令
无需改路径；该次迁移后的容量是历史快照，当前容量以上表为准。

可重新下载的pip/Conda缓存适合直接清理；Conda环境、Git工作树和频繁产生大量
小文件的缓存不迁到NTFS。

## 速度边界

4TB盘是机械硬盘，顺序写入录制数据、HDF5、视频以及长期保存checkpoint很合适，
但随机读取大量小PNG会明显慢于NVMe。模型首次加载也更慢，加载进内存/GPU后影响
通常很小。raw和DP checkpoint在`/data`，当前π0.5 checkpoint实际仍在仓库`checkpoints/`即NVMe；
不要把长期归档建议误认成已经迁移。训练数据使用HDF5或分片格式。只有
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

## 只读挂载的一键修复

NTFS卷未干净卸载时，Linux可能把 `/data` 降级为只读。先退出位于 `/data` 的终端并
停止正在读写该盘的采集或训练，然后运行：

```bash
ur5e-storage-repair
```

该命令从 `/etc/fstab` 读取设备UUID和systemd单元，正常卸载、运行
`ntfsfix --clear-dirty`、恢复
automount，并验证底层NTFS最终为 `rw`；会自动通过 `sudo` 请求一次系统权限。它不改写
fstab，也不使用强制或lazy卸载。若提示busy，先用 `fuser -vm /data` 找到占用进程并
正常退出后重试。若 `ntfsfix` 明确提示Windows休眠状态或要求 `chkdsk`，需回Windows
完整关机并检查卷，不能在Linux侧强行继续。
