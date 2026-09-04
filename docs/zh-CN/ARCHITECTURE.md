# 架构

[English](../ARCHITECTURE.md)

依赖方向有意保持单向：

```text
examples / CLI
      |
采集、数据、重播、RoboTwin适配
      |
控制层
      |
硬件层

RoboTwin适配层 ----> 被忽略的固定版本RoboTwin工作树
核心硬件层      -X-> RoboTwin
```

硬件层和数据层永远不导入 RoboTwin，因此没有ML依赖或GPU时仍能单独检查设备。
只有策略适配层可在运行时导入已经bootstrap的上游策略代码。

raw schema 保留历史上已经工作的链路：RTDE与夹爪CSV通过run ID关联双相机目录和
时间对齐CSV；转换后生成RoboTwin风格 `episode*.hdf5`。真机是一臂双相机，因此
左臂和左腕观测是明确的兼容占位，映射位于 `src/ur5e_real/data/schema.py`。

## 运行时边界

核心使用物理语义，不使用 RoboTwin 双臂字段作为内部接口：

```text
RTDE + 夹爪 + 相机
          |
          v
Observation(tcp_pose[6], gripper[1], images, timestamps)
          |
          v
RoboTwin策略适配层  <---->  固定上游工作树
          |
          v
ActionChunk(tcp targets[N,6], gripper[N], dt)
          |
          v
TCP速度限制 -> 10 Hz到500 Hz插值 -> RTDE servoJ + 串口夹爪
```

只有适配层可以把单臂7维状态编码为上游兼容向量。当前14维布局为
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`；HDF5历史组名
`joint_action` 不代表这6个数变成UR关节角，它们仍是TCP位姿。DP训练前必须用
数据往返测试已固定这一约定。

策略不拥有socket、串口或相机，只返回chunk；真机执行器负责速度限制、插值和下发。
调试顺序见 [`ROADMAP.md`](ROADMAP.md)，上游位置与后端
选择见 [`ROBOTWIN_INTEGRATION.md`](ROBOTWIN_INTEGRATION.md)。
