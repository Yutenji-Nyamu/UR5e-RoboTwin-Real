# 本轮审查与后续步骤

更新：2026-10-06。重播优先项已交付，用户要求继续模型接入；本轮补齐Metis服务与真机客户端入口。

## 已核实状态

| 环节 | 状态 | 尚缺 |
| --- | --- | --- |
| 深度采集 | `--depth` 已实现，双相机600组落盘/1200张深度回读通过 | 现场完整RGB-D示教验证 |
| Metis数据转换 | 562窗口，318训练/244验证；相对图像索引、整轨迹划分、home/初始夹爪/TCP/关节范围/数据身份均已实现 | 更多成功示教与泛化评估 |
| Metis训练 | Action/proprio首版循环；原生tiny训练/采样/重载已测 | 正式Alpha与Wan VAE权重；正式加载、单轨迹过拟合、留出评估；定期checkpoint/resume |
| Metis离线推理 | 图像+q/g→50步joint7；支持RPC与NPZ；VAE资产哈希随checkpoint校验 | 正式权重误差/延迟/显存测试 |
| Metis真机推理 | **软件入口已实现**：serve/run/home；offline/shadow/execute、live RGB和joint执行器；中间open不停车 | 正式模型的硬件shadow及现场运动验收，尚未执行 |

不需要重新研究整体架构，沿用π0.5分层即可。复用相机/RTDE观测、joint chunk和500Hz执行器；
新的模型服务负责图像编码、状态归一化和Action采样。不能直接把Metis输出文件传给现有π0.5命令，
其握手、数据集和checkpoint验证包含π0.5专属字段。

剩余顺序：

1. 下载已核对形状的官方Alpha（24.81GB）及Wan VAE（2.82GB），或使用用户提供的本地资产。
   当前Alpha loader不接受Metis stage2/stage3 DCP；如果用户提供这种格式再补loader。
2. 正式权重加载、单轨迹过拟合、完整留出评估；跑长任务前补定期checkpoint/resume。
3. 现场shadow测端到端耗时。超过1.5秒不能直接沿用同步执行；先降低采样轮数/优化或设计异步执行，再验证。
4. 回到示教home，现场小chunk执行验收。自动成功判断后续单独做；目前明确按chunk数停止。

操作命令及下载来源见 [OPERATIONS](OPERATIONS.md)。

## 本轮修正

- 新增 `ur5e-metis` 命令（已注册到RoboTwinSimReal环境）；客户端复用π0.5有界传输，独立joint7契约与请求ID。
  1.5秒超时含发送和接收；错误/过期回复断开。服务预热后监听localhost:8006；tiny不能通过production加载入口。
- home默认只打印计划；run默认offline。shadow读取RGB/RTDE、不打开串口和运动控制器；execute先做live请求验证及home校验。
  夹爪支持无限周期，但动作循环以用户指定的有限chunk数结束，默认1；不在首次open后停车或强制结束时开爪。
- 官方Alpha权重头部259272字节与meta-device模型比对：Video825、Action824、proprio2个张量全部同名同形；
  [记录](evidence/production_weight_shapes_20261006.json)。这是形状兼容证据，尚未下载权重正文、加载训练或测完整模型速度。

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
回归结果与范围见 [本轮验证记录](evidence/runtime_validation_20261006.json)。
新增测试包含真实WebSocket RGB传输、原生tiny Action经RPC输出、契约拒绝、超时、响应ID/形状错误、
可搬迁数据、验证集统计隔离、mock执行保留末次close、异常停止，以及客户端无Torch/JAX导入。
