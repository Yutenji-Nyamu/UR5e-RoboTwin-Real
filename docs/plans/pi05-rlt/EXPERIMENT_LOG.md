# RLT 实验与续接日志

## 2026-09-10：软件接入，未正式开训

- 用户授权：实现、简要检查、推送；此前确认逐阶段验收，Token首段500更新/100诊断，真机每轮4局。
- 基线：`joint5_sft_20260909_01:1000`；数据 `ur5e/pick_place_cube_joint_5_v20260908`，5条/414观测。
- donor：`d3acd650869d376c14c1406d90865ef90d680432`；原样AR token、窄适配MLP/AC算法，本地同步coordinator。
- 新隔离环境 `.venv/rlt` 已安装并pip check；`ur5e-rlt`已生成在硬件Conda环境，系统短命令软链接已安装。
- 已初始化 `cube_rlt_20260910_01`，run_id `f7740f282f400a214686b4928afacf53fd5e873e50bec5536c3b8cc3c782b7d3`。
  状态cache，selected/latest token/head均空；没有缓存414份特征、没有正式训练恢复点、没有真机round。
- 软件检查：RLT专测21通过；全仓库 **132 passed、3 skipped**；Ruff、shell语法与diff检查通过。
- 原生dummy和成功SFT固定噪声parity最大差均为0；SFT prefix768×2048，512有效image位置。
  单次热特征前向约0.164秒。报告保存在本机 `logs/rlt_checks/native_dummy.json`、`native_sft.json`，不入Git。
- 新Torch环境的非零梯度合成小模型检查通过，BC loss约0.00238、grad norm约0.0702；这是运行时检查，
  不是Token训练、BC验收或性能曲线。正式token全尺寸显存尚未实测。

下一步：读取[操作说明](USAGE.md)，确认配置后缓存、几步全尺寸显存检查，再正式Token500更新/100诊断。
Token通过才进入离线BC；首次真机reference前统一成功区域口径，首次online前复核joint/gripper探索幅度。
本轮没有物理动作、没有启动正式训练，也没有把假环境的4局结果计为任务成功率。

## 后续每次追加的摘要

保留阶段、输入checkpoint/数据范围、实际更新数与计数比、关节/夹爪/Token消融/TD/Q指标、
4局人工结果、输出checkpoint、下一步和依据。原始运行过程自动写在run内 `DECISIONS.md / decisions.jsonl`；
助手复核时把简要结论追加本文件并推送，不提交大模型、图像、原始轨迹或机器私有路径。
