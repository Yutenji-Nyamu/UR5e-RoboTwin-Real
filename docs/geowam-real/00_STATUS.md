# GeoWAM 真机：2 机实施入口
工作目录：/data/chenyiteng/projects/geowam-real
代码分支：codex/geowam-real-training
范围：2 机 GPU 6/7，数据处理、训练、离线推理；真实机器人后续迁移。

## 当前进度
- 真机代码已拉取，源提交 5c34c39bfa3e69a57b78451d9ab141a6228be3b5。
- 私有数据与稿件仓库的 Git 元数据已拉取，内容检出继续进行。
- 真机仓库包含 Metis 的旧快照；本轮仍以 Gitee 最新代码为审阅基准，正在恢复登录访问。
- GPU6/7 的 RLT owner、driver 和 Ray actors 已按身份退出，checkpoint 保留；低优先级自动回退入口停用。
- 后续文档：数据盘点、RGB-D→mask/flow→Track-UVD、模块初始化与训练/部署。
- 大数据、模型、缓存、运行输出位于同级 data/models/cache/runs，不进入代码 Git。
- Overleaf 本轮修改为零。

## 已确认的接口差异
现有真机适配器为 RGB→Action、joint7、10 Hz/H50、80维槽位10–16。
Metis 完整三专家配置为 Video/Track/Action、33帧stride4、H32；需要单独的数据适配。
指定 Alpha-Franka 权重使用 EEF10 槽位0–9，动作和归一化需按 UR5e 数据重新接入。
已确认初始化支持 Alpha Video/Action/proprio + Track 从 Video 插值，新读取模块按代码初始化。
