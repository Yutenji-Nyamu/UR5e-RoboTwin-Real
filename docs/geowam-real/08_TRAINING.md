# 训练与离线拟合

配置：`configs/geowam/train_rgbd72.yaml`。入口：`scripts/geowam/train_real.py`。

## 训练数据

五任务等权采样；全部 72 条合计 2,405 个有效窗口。每个窗口含 9 帧双视角 RGB latent、9 帧 head Track latent、当前 head RGB/depth/mask 条件、当前机器人状态、未来 32 步动作以及角色/位移/进度辅助标签。

## 检查点与拟合观察

每小时保存模型及优化器；保留最近 3 个完整检查点。保存完成以 `complete.json` 为准。每次保存后使用每任务固定窗口进行原生异步采样：条件只使用当前图像、深度、mask、状态和任务指令；未来记录用于计算拟合误差。

每个任务分别保存 6 个关节及夹爪的全 32 步曲线、关节 MAE（rad）、夹爪准确率、预测/记录视频与 Track latent。此处衡量训练集拟合；闭环成功率在真机执行阶段采集。

## 启动顺序

1. 数据和权重全部恢复并完成哈希检查。
2. 掩码、光流、UVD 和 VAE 缓存逐段检查，生成固定清单。
3. 两 GPU smoke：比较每卡 batch 1/2 与显存，检查 finite loss/gradient、checkpoint 保存/恢复、采样输出。
4. 确定并行配置并启动 30,000 步主训练；记录实际吞吐和运行时间。

数据与训推契约已通过验收；正式运行写入 `runs/train_rgbd72_v1`。每卡 batch 2、累积 1，2 卡全局 batch 4。完整 smoke 与恢复记录见 `smoke_receipt.json`。

## 本轮训练设置

30,000 步；Video lr 1e-6（冻结 500 步、再渐增 1,000 步），Track lr 1e-5（渐增 500 步），Action lr 2e-5（冻结 100 步、渐增 500 步），读取模块 lr 1e-4，Proprio lr 3e-6。第 3,000 步开始 cosine 衰减到 0.1 倍；前 3,000 步由 dense 读取过渡到 compact。保留作者的多模态噪声混合、模态 dropout 与辅助目标。

GPU 使用固定 UUID，FSDP2 block、BF16、梯度检查点、reshard_after_forward=False。短测稳定时间 2.742 秒/步、allocated 显存峰值约 64.223 GB/卡。正式吞吐在逐步日志中更新。

启动器持有本工作区锁，每 10 秒记录 heartbeat。退出异常时最多 3 次启动尝试，恢复自最近完整检查点；每小时额外执行健康与拟合检查。代码快照、初始化报告、输入契约和最终数据清单与运行输出一起保存。

## 正式启动验收

2026-10-10 03:58（北京时间）启动。第 100 步后 Action 与 Proprio 学习率按计划逐渐打开，早期约 1.8–1.9 秒/步，allocated 显存峰值 64.23 GB/卡，nvidia-smi 约 70.8 GB/卡。实际总时长还包括每小时保存与五任务采样。原生采样及 CPU VAE 解码已用 smoke 检查点验证；初始预测保存为后续拟合对照。
