# MetisWAM4D 接入记录

更新：2026-10-06。本轮实现可选深度采集、单臂数据转换和 RGB → Action 首版接口。
采用用户确认的方向：以本项目 π0.5 关节约定为准，只更新 Action 及 proprio 投影；
Track4D 和未来视频生成暂不启用。手动重播、现有 π0.5 流程独立。

## 已交付

| 内容 | 结果 |
| --- | --- |
| 源码 | 用户下载的 ZIP 已提取到 `.third_party/MetisWAM4D`，341 个普通文件；原 ZIP 另存 `.third_party/MetisWAM4D.source.zip`。无需维护 Gitee remote。 |
| 深度采集 | `ur5e-collect block_drawer_close --note "1005" --depth`；默认 RGB 路径不变。 |
| 硬件验证 | 双 D435i 60 秒、600 组 RGB-D，1,200 张深度 PNG 无损回读通过；未连接机器人。 |
| 数据转换 | 两条 success 示教生成 562 个 10 Hz/H50 窗口；保留完整任务、末次 close，padding 不计入 loss。 |
| 模型切口 | 冻结 Video 作为当前 RGB 条件，训练 Action/proprio；物理 joint7，内部有效槽位 10–16，外部可转 π0.5 joint14。 |
| 模型验证 | 原生 tiny 模型 CPU、CUDA BF16 训练/采样/保存重载通过，冻结哈希相同，图像扰动影响动作输出。 |
| 正式训练与离线推理 | `prepare / train / infer / smoke` 命令已实现；正式权重与 Wan VAE 尚未提供，因此 production 权重加载、完整微调及效果尚未验证。 |
| 真机策略执行 | 未连接机器人。后续接现有 joint executor，须补服务协议、shadow 延迟测量、显式结束条件及现场测试。 |

- [深度格式、故障处理与实测](DEPTH_RECORDING.md)
- [模型输入输出、运行命令与后续切口](MODEL_ADAPTATION.md)
- [源码清单与哈希](evidence/source_snapshot_20261006.json)
- [连续 RGB-D 落盘实测](evidence/rgbd_recording_20261006.json)
- [数据转换审计](evidence/dataset_conversion_20261006.json)
- [CPU smoke](evidence/model_smoke_cpu_20261006.json) / [CUDA BF16 smoke](evidence/model_smoke_cuda_bf16_20261006.json)
- [原始数据盘点](evidence/data_audit_20261006.json)

## 源码出处和本地状态

仓库 `droliven/MetisWAM4D_260921`，默认分支 `master`；网页观察到
`ceb1f84dc5c13d4eb759c689988d7fb505e05af0`。ZIP 没有 Git 元数据，以归档 SHA256 为实际可复核标识：
`c3dfae61227a54ee2523bb394588b79dd9a756011c9c5f3b69e53f87f6326f7a`。
不把网页提交号冒充已验证的 checkout HEAD。

归档另有 24 个指向作者服务器 RoboTwin 仿真资产的绝对软链接，本机无法解析，未创建；
清单保存在本地 `UNRESOLVED_EXTERNAL_LINKS.json`。本轮采用的 `metiswam4d/` 模型源码不依赖它们。
未使用另一个 `metiswam4d_inspired_by_internw0/` 实现，也未修改上游源码。
用户已明确要求两套源码公开跟踪：`.third_party/MetisWAM4D` 和 `.third_party/RoboTwin`
均作为主仓库普通文件提交；权重、原始图像、转换产物和ZIP仍忽略。
最新范围与待办见 [本轮审查与后续步骤](STATUS_AND_NEXT.md)。

本轮验证：项目测试 **189 passed, 3 skipped**；其中 Metis/采集/RGB-D 定向测试 25 passed。
跳过的是需显式开启的本地RPC测试，详见测试运行输出；这些数字不表示 production 模型或真机推理已经验收。

## 接下来缺什么

1. 匹配模型配置的 OpenWAM-Alpha `.safetensors`（Video/Action/proprio）和 Wan2.2 TI2V-5B VAE。
   当前加载入口针对 Alpha；若提供的是 Metis stage2/stage3 DCP，另补对应 checkpoint loader。
2. 用正式权重跑一条轨迹过拟合和留出轨迹评估，测显存、推理延迟、动作限幅情况。
3. 将新策略接到现有真机执行边界。数据里的 `close → open → close` 本身可正常学习；
   要调整的是旧抓放执行器“open 后任务结束”的规则，首版数据转换已不受该规则限制。

文档随实际验证更新，区分相机组件实测、mock 采集回归、tiny 结构测试、正式模型和现场执行。
