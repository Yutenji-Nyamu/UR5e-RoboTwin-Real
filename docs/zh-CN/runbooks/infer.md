# UR5e 真机推理

[English](../../runbooks/infer.md)

1. 运行硬件doctor，将机械臂放到验证过的起点。
2. 将PolyScope切到Local模式。
3. 使用现有URP启动机器人端servoJ循环，或依据
   `robot_programs/servoj_control_loop.script` 创建程序。
4. 先不运动，仅检查路径和任务配置：

```bash
ur5e-real infer-act --config configs/lab.yaml \
  --task pick_block_bowl --task-config simple --episodes 15 \
  --checkpoint policy_best.ckpt
```

5. 工作区清空且人员守在急停旁后，才添加 `--execute`。

推理将头相机映射到 `cam_high`，腕相机映射到 `cam_right_wrist`，并为未使用的
`cam_left_wrist` 复制腕图。单物理臂复制到ACT左右臂槽，未使用的左夹爪保持0。

ACT通过RTDE接收实测TCP/运行状态，并通过RTDE输入寄存器写入TCP设点；运行中的
PolyScope循环负责IK和servoJ。这不是socket `movel`重播链路。
