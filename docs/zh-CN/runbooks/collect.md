# 采集流程

[English](../../runbooks/collect.md)

## 日常固定流程

开始前：机器人、夹爪和双相机上电；PolyScope切到 **Remote Control**；确认夹爪
没有夹持物体，清空回原位路径和工作区，人员守在急停旁。

第一条，初始化采集现场：

```bash
ur5e-collect-init
```

该命令内部自动使用 `RoboTwinSimReal` 环境和本仓库配置，依次检查全部硬件、低速
回到固定原位、RTDE确认到位并打开夹爪。看到 `[READY]` 才继续。

第二条，开始一条遥操作采集：

```bash
ur5e-collect pick_block_bowl
```

`pick_block_bowl` 是本条轨迹的任务名。命令启动双相机预热、RTDE记录和
freedrive；随后可以自由拖动机械臂：

- `c`：闭合夹爪；
- `o`：打开夹爪；
- `q`：正常结束；
- `Ctrl+C`：中断结束。

停止后输入 `s`、`f` 或 `a`，分别标记成功、失败或中止；直接回车可稍后标记。
需要记录场景变化时使用：

```bash
ur5e-collect pick_block_bowl --note "red block, trial 1"
```

## 输出

2026-09-08 采集器更新后，新轨迹使用 **raw schema v3**，同时记录实测关节与 TCP。
采集命令和按键不变。进入 freedrive 前先验证一个完整 RTDE 数据包；缺少关节字段、
维度错误或出现 NaN/Inf 会直接报错，不填零继续录制。启动时会显示：

```text
[STATE] schema=3 actual_q[6] + actual_qd[6] + TCP[6]; same RTDE packet
```

`rtde_tcp_gripper_<run_id>.csv` 保留旧文件名及前9列，追加 `actual_q_0`～`actual_q_5`
（弧度）、`actual_qd_0`～`actual_qd_5`（弧度/秒）和 `host_receive_time_s`（Unix秒）。
关节角是实测值，不是 IK 估计或目标指令。manifest 记录字段来源、关节顺序、单位，
以及 freedrive 前的初始 q/TCP/TCP offset 快照。主机接收时间不是相机曝光时间，
不代表相机与机器人硬件严格同步。

最后按 `o` 后至少继续记录1秒，直到实体夹爪完成释放。结束时
`quality.last_open_to_last_frame_s` 记录图像尾段跨度，不足或缺失会提示；不因此
延迟 `q`/Ctrl+C 停止。该指标不是成功判定，也不代替逐帧缺失检查。
原始文件不裁剪；按开爪事件保护导出尾段留待后续数据处理开发。

原始数据位于 `/data/robotics/ur5e-real/raw`。每条轨迹有一个
`session_<run_id>.json`，记录任务、时间、时长、结果、代码提交、计数和所有产物
路径；图像、RTDE和夹爪CSV不会被后续复核改写。

列表新增 `SCHEMA` 和 `STATE`：旧 v1/v2 为 `tcp`，新双记录且非空为 `joint+tcp`，
零样本为 `none`。现有 DP/ACT 转换与重播仍读取 TCP，新 CSV 也兼容；此次采集升级
不等于已经支持关节策略训练或关节执行。

查看最近20条：

```bash
ur5e-real sessions --config ~/UR5e_RoboTwin_Real/configs/lab.yaml
```

底层逐项排错见[硬件逐项调试](hardware_commissioning.md)。
