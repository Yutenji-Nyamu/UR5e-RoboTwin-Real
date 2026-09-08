# 采集双记录开发交付：2026-09-08

本轮按用户最新要求，只开发采集、测试和日志。不启动补采，不实现 joint executor、
π0.5 训练/推理或 RLT。当前 [决策入口](DECISIONS.md) 与 [采集说明](../../zh-CN/runbooks/collect.md)。

## 已实现

- [hardware/rtde.py](../../../src/ur5e_real/hardware/rtde.py)：新增 `RtdeStateClient` 与
  不可变 `RtdeRobotState`，同包读取 timestamp、actual_TCP_pose、actual_q、actual_qd、
  tcp_offset。六维字段和时间必须为有限数值；保留关节分支，不做取模或 IK。
- 新 `RtdeStateCsvWriter` 保留旧CSV前9列，追加 q6、qd6、Unix接收时间；使用独占创建。
  旧 `RtdeTcpClient.receive()` 和旧 CSV writer 调用方式保持兼容。
- [collection/session.py](../../../src/ur5e_real/collection/session.py)：raw schema v3，
  初包验证在 freedrive/夹爪串口操作之前；manifest 保存单位、关节顺序、字段来源、
  pre-freedrive 初始状态及 TCP offset。每个已接收状态即使缺相机图像也落盘。
- 同名 session 拒绝覆盖；控制器时间不递增则结束为 failed；图像写入失败不计为成功帧对。
  原有人工 outcome、重复 close/open 事件与清理流程保留。
- 结束时保存最后 open 到最后图像的时间跨度；不足1秒提示，不延迟退出、不自动标成功、
  不修改原始尾段。此指标只衡量尾段跨度，不证明物理释放或中间没有丢帧。
- [session_manifest.py](../../../src/ur5e_real/data/session_manifest.py)：sessions 列表新增
  schema 和 recorded state，区分旧 `tcp`、新 `joint+tcp`、零样本 `none`、未知 `unknown`。
- 数据管理中英说明及数据盘 `DATA_LOG.md` 记录格式分界：升级前29份动作CSV均无实测q，
  最新 run 为 `20260907_174521`；没有新增真实 v3 轨迹。

## 验证证据与边界

- 在现有 `RoboTwinSimReal` 环境运行全量测试：**57 passed**；Ruff 检查通过。
- 新增19项测试，包含单包字段/单位值保存、NaN/缺字段拒绝、完整模拟采集、初包/中途失败、
  Ctrl+C、重复夹爪事件、相机缺帧、图像写入失败、尾段检查与覆盖保护。
- 旧/新CSV分别通过 TCP HDF5→ACT/DP 数据处理回归，且新关节列不改变旧TCP标签；
  重播读取与原 TCP recipe/API 亦有回归覆盖。
- 已检查本机现有 `ur5e-collect --help` 和只读 sessions 列表；现有命令直接加载本仓库源码，
  无需重装环境或改 shell 命令。没有连接机器人、相机或串口进行现场验证。

## 留待之后，不应写成已完成

- 真机 recipe 兼容性/实测 q 与完整现场录制验证；第一条真实 v3 run ID。
- 图像硬件曝光时间：当前接收时间是主机 Unix 时间，sync 仍是控制器时间关联，不宣称硬同步。
- 以事件保护尾段的 joint exporter、质量选择、joint controller/home q、π0.5 和 RLT。
  目前 raw 双记录已开发；现有 TCP 模型不会自动变成 joint 模型。

用户已明确：本轮专注开发，上述现场和模型事项在开发交付之后再讨论。
