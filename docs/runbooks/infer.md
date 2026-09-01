# Infer on the real UR5e

1. Run the hardware doctor and move the arm to a validated start pose.
2. Switch PolyScope to Local mode.
3. Start the robot-side servoJ loop using the existing validated URP or create a
   program from `robot_programs/servoj_control_loop.script`.
4. First validate paths and task configuration without motion:

```bash
ur5e-real infer-act --config configs/lab.yaml \
  --task pick_block_bowl --task-config simple --episodes 15 \
  --checkpoint policy_best.ckpt
```

5. With the workspace clear and a person at the emergency stop, add `--execute`.

Inference maps the physical head camera to `cam_high`, the wrist camera to
`cam_right_wrist`, and duplicates the wrist image for the unused
`cam_left_wrist` channel. The single physical arm is duplicated into the ACT
left/right arm slots; the unused left gripper remains zero to match conversion.
