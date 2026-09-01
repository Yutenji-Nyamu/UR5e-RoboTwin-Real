# 数据采集

[English](../../runbooks/collect.md)

1. 清空工作区，人员守在急停旁。
2. 确认当前freedrive流程所需的机器人模式。
3. 运行 `ur5e-real doctor --config configs/lab.yaml --hardware`。
4. 运行 `ur5e-real collect --config configs/lab.yaml`。
5. 按 `c` 闭合夹爪、`o` 打开、`q` 停止。
6. 检查session manifest与两个相机目录后再转换。

raw输出写入配置的 `data_root/raw`，不进入Git。`--save-video` 可额外生成便于预览
的MP4；同步转换仍以PNG帧为来源。通过验收的raw run只追加、不原地修改，详见
[`../DATA_MANAGEMENT.md`](../DATA_MANAGEMENT.md)。
