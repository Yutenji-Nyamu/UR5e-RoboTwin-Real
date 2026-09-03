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

原始数据位于 `/data/robotics/ur5e-real/raw`。每条轨迹有一个
`session_<run_id>.json`，记录任务、时间、时长、结果、代码提交、计数和所有产物
路径；图像、RTDE和夹爪CSV不会被后续复核改写。

查看最近20条：

```bash
ur5e-real sessions --config ~/UR5e_RoboTwin_Real/configs/lab.yaml
```

底层逐项排错见[硬件逐项调试](hardware_commissioning.md)。
