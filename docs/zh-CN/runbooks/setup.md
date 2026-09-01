# 环境设置

[English](../../runbooks/setup.md)

1. 创建或激活 `RoboTwinSimReal`，以editable方式安装本仓库。
2. 将 `configs/lab.example.yaml` 复制为被忽略的 `configs/lab.yaml`。
3. 核对B81L机器人地址、夹爪稳定by-id路径和两个RealSense序列号。
4. 确认用户属于 `dialout` 与 `robotdata`；组变更后重新插拔串口或重新登录。
5. 确认 `/data/robotics/ur5e-real` 已挂载且可写。
6. 运行 `ur5e-real doctor --config configs/lab.yaml`。
7. 设备接好后加 `--hardware` 再运行一次。

无法激活shell环境时，自动化使用 `conda run -n RoboTwinSimReal ...`。本机Conda
绝对安装路径只属于本机工具，不写进公开仓库。
