# 采集与重播速查

[English](../../runbooks/operator_workflows.md)

这是日常操作的固定入口。命令已封装仓库目录、`RoboTwinSimReal` 环境和实验室配置，
无需先执行 `cd` 或 `conda activate`。

## 共通准备

- UR5e、夹爪和两台相机上电，相机线与夹爪线已连接；
- PolyScope 切换到 **Remote Control**；
- 清空机械臂回原位及后续运动路径，人员守在急停旁。

## 流程一：采集

1. 初始化：

   ```bash
   ur5e-collect-init
   ```

   自动检查设备、低速回固定原位、由 RTDE 确认到位并打开夹爪。看到 `[READY]`
   后继续。

2. 开始遥操作采集：

   ```bash
   ur5e-collect <任务名> --note "可选的场景备注"
   ```

   示例：

   ```bash
   ur5e-collect pick_place_cube --note "red cube, trial 1"
   ```

3. 看到 `[READY] recording and freedrive are active` 后开始拖动机械臂：

   - `c`：闭合夹爪；
   - `o`：打开夹爪；
   - `q`：结束本条采集；
   - 结束后输入 `s`、`f` 或 `a`：标记成功、失败或中止。

终端中的 `[RUN] <RUN_ID>` 是本条数据编号。它由时间自动生成，只需记下，不需要
手工填写。数据保存到 `/data/robotics/ur5e-real/raw`。

## 流程二：重播

1. 清空回原位路径，然后初始化：

   ```bash
   ur5e-replay-init
   ```

   自动检查设备、低速回固定原位并打开夹爪。看到 `[READY]` 后继续。

2. 按录制时的起始位置布置物体，确认整条轨迹空间净空。

3. 重播指定数据：

   ```bash
   ur5e-replay <RUN_ID> --execute
   ```

   示例：

   ```bash
   ur5e-replay 20260903_182752 --execute
   ```

程序会低速对齐轨迹起点，然后完整执行机械臂路径和夹爪事件。第一次重播未知轨迹
时，可先去掉 `--execute` 查看摘要；更完整的分段验证方法见[重播流程](replay.md)。

## 两条流程的边界

采集与手动重播已经形成独立闭环。这里的重播使用 socket `movel`，用于验机和物理
回归；后续 RoboTwin 策略训练与 Diffusion Policy 真机推理使用独立的 RTDE
`servoJ` 执行链路。
