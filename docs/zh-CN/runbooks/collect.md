# 录制与数据登记

[English](../../runbooks/collect.md)

## 每次开机后

清空工作区、人员守在急停旁，确认机器人和夹爪上电、两台相机接好，然后：

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
ur5e-real doctor --config configs/lab.yaml --hardware
```

全部为 `OK` 后再录制。该检查只读取状态，不运动机械臂或夹爪。

## 录一条原始轨迹

PolyScope 切到 Remote Control，并确认可以进入 freedrive：

```bash
ur5e-real collect --config configs/lab.yaml --task pick_block_bowl \
  --initial-gripper closed
```

`--initial-gripper` 只登记开始录制时两指的实际状态，不会驱动夹爪；按现场情况填写
`open` 或 `closed`。需要记录场景变化时可加 `--note "red block, trial 1"`。录制中：

- `c`：闭合夹爪；
- `o`：打开夹爪；
- `q`：正常结束；
- `Ctrl+C`：中断结束。

每条轨迹生成一个 `session_<run_id>.json`，自动记录任务、起止时间、时长、停止
方式、代码提交、样本/事件/图像对数量和所有原始产物路径。原始数据位于
`/data/robotics/ur5e-real/raw`。

## 录完后标记结果

使用采集命令最后打印的 session 路径：

```bash
ur5e-real review /data/robotics/ur5e-real/raw/action/session_RUN_ID.json \
  --result success --note "clean demonstration"
```

也可标为 `failure` 或 `aborted`。再次 review 会追加一条历史记录；不会改动图像、
RTDE或夹爪CSV。查看最近20条：

```bash
ur5e-real sessions --config configs/lab.yaml
```

`--save-video` 只额外生成便于查看的MP4；同步转换仍以PNG帧为准。原始数据的保留
规则见 [`../DATA_MANAGEMENT.md`](../DATA_MANAGEMENT.md)。
