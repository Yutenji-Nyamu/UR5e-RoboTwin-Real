# 真机集成路线图

[English](../ROADMAP.md)

状态：2026-09-04。设备、采集、重播、DP离线链路和shadow已通过，下一步是servoJ
小步验证。RoboTwin 始终位于窄策略适配层之上。

## 固定原则

1. 硬件、时间戳、安全和执行归本仓库负责。
2. RoboTwin 负责模型与训练语义。固定版本 DP 基线（horizon `8`、观测 `3` 步、
   动作 `6` 步、`10 Hz`）默认不改；只有实测证据和明确决策才能改变。
3. 手动录制/重播与学习策略执行是两条独立运动链路。
4. 首次物理输出不能与其他设备测试合并；任何运动都需要人员在现场。

## 边界

```text
设备 → 规范采集 → 数据适配 → RoboTwin 训练/checkpoint
设备 → 在线 Observation → 策略适配 → ActionChunk → 受保护执行器
```

核心契约使用单臂、物理语义：

- `Observation`：控制器/单调时间戳、TCP位姿 `[6]`、夹爪状态、头/腕图像、
  有效性和标定身份；
- `ActionChunk`：`N x 6` 绝对TCP目标、`N`个夹爪目标、步长、生成时间和过期时间。

只有策略适配层可以编码为 RoboTwin 双臂向量。当前14维兼容布局初始映射为
`[tcp(6), dummy_gripper, tcp(6), physical_gripper]`。

## 当前证据

| 底层轮子 | 状态 | 下一物理门 |
|---|---|---|
| UR5e 网络/Dashboard | 连通；PolyScope 5.13，`RUNNING`，安全 `NORMAL` | 输出前由人确认模式 |
| RTDE 读取 | 10 Hz采集已用于完整session | servoJ时复测500 Hz反馈 |
| 串口夹爪 | 稳定by-id路径；打开/闭合均已实测 | 策略夹爪解码 |
| 双 RealSense | 双路采集、60帧预热和DP shadow已实测 | 正式数据采集 |
| socket 重播 | session `20260903_182752` 已完整重播成功 | 保持为独立回归链路 |
| RTDE servoJ | ACT 客户端、recipe、机器人程序均存在 | 保持、小步、斜坡、watchdog |
| Diffusion Policy | 227步Zarr、原生batch、GPU训练、checkpoint加载、offline和shadow已通过 | servoJ后再做实机执行 |

## Chunk 执行

RoboTwin DP 输出6步动作。第一版按上游顺序完整执行6步，再进行下一次推理；不默认
改成只执行第1步，也不先加入chunk重叠融合。

每个 chunk 到达机器人前依次经过：

1. 旋转向量连续化和单步速度限制；
2. servoJ 的10 Hz到500 Hz插值；
3. RTDE运行状态和实测TCP反馈。

完整6步基线实测后，再决定是否需要额外日志、提前重规划或chunk融合。详细语义见
[`DIFFUSION_POLICY_PLAN.md`](DIFFUSION_POLICY_PLAN.md)。

## 运动后端

- `SocketMovelReplayBackend`：Remote模式、批量 `movel`、开环计时；仅用于手动
  重播。
- `SocketSpeedLPolicyBackend`：RTDE读状态、10 Hz socket `speedl`写动作，目标EMA
  可调；学习策略首次验机后端。
- `RtdeServoJBackend`：Remote模式自动启动机器人程序、500 Hz RTDE设点与反馈；
  明确的实验后端。

两者绝不自动切换，详见 [`ROBOTWIN_INTEGRATION.md`](ROBOTWIN_INTEGRATION.md)。

## 验收门

### 0 — 离线契约

- 增加 `/joint_action/vector`、DP Zarr适配和往返测试。
- 用合成 episode 测试时间对齐、旋转连续性和插值。

退出条件：一个规范episode可生成正确DP batch，并解码回相同单臂语义。

状态：已完成。

### 1 — 只读设备

- Dashboard/RTDE状态、稳定串口路径、双相机出流、可写数据根目录。

退出条件：所有独立检查通过。当前配置已完成此门。

### 2 — 有边界的设备输出

- 机械臂停稳后测试夹爪打开/闭合。
- 扶住机械臂测试 freedrive 开启/停止。

退出条件：准确观察并记录命令与停止行为。

状态：已完成。

### 3 — 手动录制/重播

- 录制短路径，检查 manifest/时间，dry-run，只执行一段，再比较录制和重播的
  TCP/夹爪轨迹。

退出条件：无位姿跳变，跟踪/时间误差有界；此门与ML独立。

状态：已完成一次完整采集和重播。

### 4 — 策略运动后端

- 先用一个模型chunk确认socket speedL；servoJ另行测试保持、毫米级小步、慢速
  斜坡、跟踪和停止延迟。

退出条件：安全限制以及实测频率/延迟通过。

### 5 — 数据与DP离线

- 采集一个任务episode；检查图像/时间/动作；转换DP；单episode过拟合；可视化
  6步预测；测量推理延迟。

退出条件：不输出机器人命令时，可复现加载checkpoint且chunk数值合理。

状态：已完成转换、训练、加载和shadow管线验证；当前模型不用于任务效果评价。

### 6 — Shadow再实机

- Shadow：输入真实传感器，只记录预测，不下发。
- 仅机械臂、保守chunk；仅在有价值时比较socket与RTDE。
- 先执行完整6步socket；RTDE保持明确A/B；最后启用带去抖的夹爪。

退出条件：试验可复现，并记录停止原因及原始/保护/实测轨迹。

状态：shadow已完成；下一步实测socket策略执行，servoJ保持独立门控。

## 人员与自动化分工

人员负责断路器、工作区、急停、Remote模式和安全起点。自动化负责启动控制程序、
诊断、配置、命令、日志、测试、图表、适配和提交。每次实际输出前，
自动化只需说明准确设备、后端和有边界动作，由现场人员确认一次。
