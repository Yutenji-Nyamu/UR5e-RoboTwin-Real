# 本轮审查与后续步骤

更新：2026-10-06。用户调整优先级：先完成重播的毫秒参数及RTDE默认，模型接入保持明确待办，暂不扩展。

## 已核实状态

| 环节 | 状态 | 尚缺 |
| --- | --- | --- |
| 深度采集 | `--depth` 已实现，双相机600组落盘/1200张深度回读通过 | 现场完整RGB-D示教验证 |
| Metis数据转换 | 两条成功轨迹生成562个窗口，10Hz/H50，保留close/open/close | 图像索引搬迁、训练/验证划分；真机初始姿态及运动边界契约 |
| Metis训练 | Action/proprio首版循环；原生tiny训练/采样/重载已测 | 正式Alpha与Wan VAE权重；正式加载、单轨迹过拟合、留出评估；定期checkpoint/resume |
| Metis离线推理 | 图像+q/g→50步joint7，兼容joint14 | 正式权重的误差、延迟、显存测试；VAE资产身份校验 |
| Metis真机推理 | **尚未实现端到端入口**，当前infer只写NPZ | server/client、live observation、offline/shadow/execute、初始化/停止契约 |

不需要重新研究整体架构，沿用π0.5分层即可。复用相机/RTDE观测、joint chunk和500Hz执行器；
新的模型服务负责图像编码、状态归一化和Action采样。不能直接把Metis输出文件传给现有π0.5命令，
其握手、数据集和checkpoint验证包含π0.5专属字段。

继续接真机的顺序：

1. 由数据与lab配置生成一致的home_q、initial_gripper、TCP offset、joint/TCP边界、dataset identity。
2. 加Metis独立服务、客户端握手和有界请求超时，接入offline/shadow；此阶段不发送动作。
3. 复用joint执行器，支持执行chunk前K步再观测；用显式结束条件，不因中途open结束。
4. 正式权重完成离线拟合/留出评估与shadow延迟测试后，再进行现场执行验收。

## 本轮修正

- 原首版把Action参数及AdamW状态也设成BF16。数值复现：1e-5更新可被BF16舍入掉。
  已修为可训练参数/优化器状态FP32、冻结Video BF16、计算autocast BF16；checkpoint保存/重载保留混合精度。
  [默认学习率下的GPU验证](evidence/model_smoke_fp32_master_20261006.json)通过，70个可训练张量更新、冻结哈希不变。
- 重播默认RTDE，新增 `--inference-ms`，默认80ms，0关闭模拟等待；原 `--chunk-gap` 秒单位仍兼容。
  只在chunk之间保持末设点，不在每动作点等待；17项测试与真实记录dry-run通过，未进行实体重播。

## 第三方代码的Git管理

用户已明确授权全部源码公开跟踪。两套源码均作为主仓库普通文件，不使用submodule：

- `.third_party/RoboTwin`：1557个文件，约14.4MB；原基线`210720340637cb4619283b295dde4cdd807c9e66`，
  保留本项目四个兼容补丁，排除17个上游误跟踪的pyc缓存。包含上游9KB固定空语言embedding小资源。
- `.third_party/MetisWAM4D`：341个文件，约5.5MB；保留下载快照。24个作者服务器绝对软链接不导入，
  模型源码不依赖这些仿真链接。
- 权重、原始数据、生成产物、ZIP和缓存不跟踪。旧RoboTwin Git历史保留在本地
  `outputs/vendor-audit/RoboTwin.git`，该备份不提交。
- `integrations/vendor_sources.lock.json`记录源文件哈希和上游出处；bootstrap改为校验随仓库提供的源码，
  π0.5 native校验同样改用已含补丁的源文件哈希。修改vendor源码后需同步更新此清单再测试。

复核命令（仅标准库，无硬件、无模型权重）：

```bash
PYTHONPATH=src python -m ur5e_real.vendor all
bash scripts/bootstrap_robotwin.sh
```

本轮没有运行模型驱动真机，也未把tiny测试误记为正式模型训练完成。
当前项目回归205 passed、3个需显式开启的RPC测试跳过；π0.5原生模型和配置导入通过。
