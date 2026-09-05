# 数据管理

[English](../DATA_MANAGEMENT.md)

目标是反复利用有价值的示范数据，同时让每次训练和评估都能追溯到不可变的源数据。

## 生命周期

1. **Raw**：RTDE、夹爪、图像和同步表只写一次；小型 session manifest 是唯一索引，
   后续只追加人工复核历史。
2. **Validated**：附加时间、帧、动作质量报告以及接受/拒绝原因。
3. **Converted**：从明确的 raw run ID 生成带schema版本的数据；可以重建。
4. **Training**：记录dataset ID、策略、上游提交、适配层提交、完整配置、seed、
   环境锁和checkpoint哈希。
5. **Evaluation**：每次试验关联唯一checkpoint及数据/配置；保存成功、停止原因、
   延迟、原始预测、受保护命令和实测状态。
6. **Archive**：保留有价值raw和精选checkpoint；只清理可复现缓存及明确损坏数据。

## 目录契约

```text
/data/robotics/ur5e-real/
├── raw/                 # 采集器写入，只追加session
├── converted/run_*/     # 可追溯到raw run ID的规范HDF5
├── converted/dp/        # RoboTwin DP Zarr
├── checkpoints/dp/      # dataset/seed/训练运行层级
├── logs/                # 验证、训练、shadow、实机评估
└── DATA_LOG.md          # 每次数据生成、裁剪、训练的一两句人工日志
```

使用可读dataset ID，例如 `pick_place_cube-simple-17-trim2mm-v20260905_150058`。manifest
记录源run ID、任务/配置、相机/标定、采样率、schema版本、转换器提交、质量结论和
校验清单。人工备注只追加，不通过重命名或改写raw文件表达。

## 一条轨迹，一个索引

每条录制轨迹只维护一个权威的 `session_<run_id>.json`。它只引用各项产物，不重复
枚举同步CSV中已有的逐帧记录。采集器会在录制结束时补齐以下字段：

- 身份：run ID、任务类型/名称、schema版本、代码提交；
- 时间：开始、结束、时长；
- 结果：`unreviewed`、`success`、`failure` 或 `aborted`，可附停止/失败原因和短备注；
- 产物：RTDE/动作CSV、夹爪事件CSV、同步CSV、头部/腕部图像目录，以及可选视频/
  预览路径；
- 计数：RTDE样本、夹爪事件、同步图像对数量；
- 来源：机器人、夹爪、相机、标定ID、采集配置和采样率。

同步CSV继续作为逐帧索引。`ur5e-real review` 在 manifest 的 `reviews` 中追加带时间
的结论，并把顶层 `outcome` 设为最新结论；它不改写已采集的传感器/动作文件。
转换时只选任务名匹配且 `outcome=success` 的run；每个HDF5 episode保留源run ID。
DP Zarr继续记录全部源run ID、schema版本、状态布局、动作语义和图像尺寸。裁剪版本还
记录源长度及每条轨迹的精确 `[start, stop)`；raw与HDF5保持不变。

## 保留规则

- 通过验收的raw示范：默认长期保留。
- 被拒绝/损坏的raw：先保留到原因复核，再归档或明确删除。
- converted：保留活跃版本；过时版本可由raw重建。
- checkpoint：保留 `best`、`last`、里程碑及指标；评估后可裁剪密集的中间步。
- 缓存和可下载的上游资产：不具权威性，空间不足时可清理。

## 迁移已有大目录

Conda、venv、Git活跃目录不迁到NTFS。数据集或checkpoint按以下流程：

1. 确认没有进程使用源目录；
2. 复制到 `/data/robotics/staging`，保留时间戳；
3. 确认复制成功，并用metadata dry run快速核对目录树、大小和时间戳；只有不可替代
   或疑似损坏的数据才做全量内容校验；
4. 将目标重命名到最终共享位置；
5. 将源目录改为回滚名，创建一个明确软链接，并运行实际读取/加载测试；
6. 通过验证后才删除回滚副本，并记录迁移。

这样既不会训练一次就丢失数据来历，也能让大文件离开NVMe根盘。
