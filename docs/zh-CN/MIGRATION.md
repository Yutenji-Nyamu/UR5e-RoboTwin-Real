# 迁移记录

[English](../MIGRATION.md)

历史仓库保持不变，仍是旧实验和旧数据的事实来源。新仓库只迁移已经工作的真机
主链路。

| 能力 | 历史来源 | 新归属 |
|---|---|---|
| URScript/freedrive | `freedrive_urscript.py` | `hardware/urscript.py` |
| 串口夹爪 | `gripper_serial.py` | `hardware/gripper.py` |
| RTDE记录 | `rtde_tcp_logger.py` | `hardware/rtde.py` |
| 双RealSense | `realsense_dual_collect_2_folder_func.py` | `hardware/realsense.py` |
| 集成采集 | `collect_data_action_arm_gripper_dual_camera_no_cv*.py` | `collection/session.py` |
| HDF5转换 | `convert_2_hdf5_output_log_dual_gripper.py` | `data/convert_hdf5.py` |
| 重播 | `replay_action_socket_batch.py` | `replay.py` |
| ACT转换/推理 | 历史 `RoboTwin/policy/ACT` 中新增文件 | `adapters/robotwin_act` |

历史 RoboTwin submodule 指向提交
`210720340637cb4619283b295dde4cdd807c9e66`，后来被转成普通目录并积累本地修改。
新仓库恢复边界：上游固定版本且不跟踪；我们的适配和补丁独立提交。

有意排除：RobotEnvironment、ReKep、CaP/RAG、API集成、raw数据、HDF5/视频、
checkpoint、缓存、驱动压缩包、重复RTDE源码、备份和已替代实验脚本。
