# Migration record

[简体中文](zh-CN/MIGRATION.md)

The historical repository remains unchanged and is the source of truth for old
experiments and data. This repository migrates only the working real-robot path.

| Capability | Historical source | New owner |
|---|---|---|
| URScript/freedrive | `freedrive_urscript.py` | `hardware/urscript.py` |
| Serial gripper | `gripper_serial.py` | `hardware/gripper.py` |
| RTDE logging | `rtde_tcp_logger.py` | `hardware/rtde.py` |
| Dual RealSense | `realsense_dual_collect_2_folder_func.py` | `hardware/realsense.py` |
| Integrated collection | `collect_data_action_arm_gripper_dual_camera_no_cv*.py` | `collection/session.py` |
| HDF5 conversion | `convert_2_hdf5_output_log_dual_gripper.py` | `data/convert_hdf5.py` |
| Replay | `replay_action_socket_batch.py` | `replay.py` |
| ACT conversion/inference | files added under historical `RoboTwin/policy/ACT` | `adapters/robotwin_act` |

The historical RoboTwin submodule pointed to commit
`210720340637cb4619283b295dde4cdd807c9e66`. It was later converted into a normal
folder and accumulated local changes. This repository restores the boundary:
upstream is pinned and ignored; our adapter and patch are tracked independently.

Excluded on purpose: RobotEnvironment, ReKep, CaP/RAG, API integrations, raw
datasets, HDF5/video output, checkpoints, caches, driver archives, duplicated
RTDE source, backups, and superseded experiment scripts.
