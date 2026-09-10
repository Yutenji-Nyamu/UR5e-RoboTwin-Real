# RLT 操作说明

状态：软件已实现；**尚未训练正式 token/actor/critic，尚未执行 RLT 真机回合**。
原成功的 `ur5e-pi05-infer` 与旧 TCP DP 命令不变。验收依据见[实施记录](IMPLEMENTATION.md)。

## 一次安装与当前入口

在仓库根目录，先按原说明装好 π0.5 和 `RoboTwinSimReal`，再执行：

```bash
bash scripts/setup_rlt_env.sh
conda activate RoboTwinSimReal
bash scripts/install_operator_commands.sh
```

本机已完成上述增量安装：新命令 `ur5e-rlt` 在硬件环境运行，自动用 `.venv/pi05` 缓存原生特征、
用 `.venv/rlt` 训练/服务 PyTorch。不向已成功的环境叠装 RLinf/Ray。
RLT 依赖锁在 `integrations/rlt/requirements.lock`；支持 Python 3.11、Torch 2.4.1/cu121。

当前仅初始化了 `cube_rlt_20260910_01`，基线为 `joint5_sft_20260909_01:1000`，状态是 `cache`。
没有启动下面的缓存或正式训练命令：

```bash
ur5e-rlt status cube_rlt_20260910_01
ur5e-rlt cache cube_rlt_20260910_01
ur5e-rlt train-token cube_rlt_20260910_01
```

后两条是下一阶段工作：先缓存5条示教的414个观测，再卸载 JAX 模型，默认只训练 token 500 更新，
每100更新诊断/保存，最后一步也保存。再次执行 `train-token --steps N` 是从最近恢复点**追加 N 更新**。
500 是首段预算，不是“达到500自动通过”；默认 batch2 仍需正式训练前几步确认显存。

新实验用 `ur5e-rlt init NEW_RUN --checkpoint RUN:STEP [--config overrides.json]`。
配置分 `token/token_train/algorithm/real/save`，完整键见[默认配置](../../../src/ur5e_real/adapters/rlinf_rlt/config.py)。
算法/动作配置一旦初始化即冻结；改表示、动作域或超参数需另建版本，当前不支持跨 run 自动迁移恢复点。
各次 `--steps` 是可调整的工作预算，不改变模型身份。`cache --limit N` 仅用于开发，不能用于真机轮次。

## 按阶段推进

训练结束不会自动启动机械臂，也不会自动判定收敛。助手读取诊断、选择 checkpoint，
用 `decide` 写明原因与证据；运行日志的 `DECISIONS.md`、`decisions.jsonl` 同时更新。

例如 token 诊断通过后（路径以实际打印为准）：

```bash
ur5e-rlt decide cube_rlt_20260910_01 --stage bc \
  --checkpoint logs/rlt/cube_rlt_20260910_01/checkpoints/token/update_00000500.pt \
  --evidence logs/rlt/cube_rlt_20260910_01/token_diagnostics.json \
  --reason "填入重建与token置零/打乱诊断的实际结论"
ur5e-rlt train-bc cube_rlt_20260910_01 --steps 100
```

这里100只是一次有界工作量示例，**不是用户已经批准的正式 BC 预算或验收阈值**。
助手检查关节误差、夹爪阈值判断、辅助状态扰动等结果，再通过相同 `decide` 入口选择 head 并进入 `reference`。
继续同阶段训练无需伪造阶段通过；重复真机轮次前必须做同阶段复核决定。

| 已接受阶段 | 本阶段命令/行为 | 下一关 |
| --- | --- | --- |
| `token` | `train-token`，VLA 不参与优化 | `decide --stage bc --checkpoint TOKEN --evidence DIAG --reason ...` |
| `bc` | `train-bc --steps N`，纯 reference 模仿 | 选 HEAD，进入 `reference` |
| `reference` | 原 π0.5 每轮4局；`train-ac --mode warmup --steps N` | 选已预热 HEAD，进入 `actor_probe` |
| `actor_probe` | 确定性小 actor 每轮4局；必要时继续 warmup | 依据真实闭环结果进入 `online` |
| `online` | 小 actor 固定 std 探索，每轮4局；`train-ac --mode online --steps N` | 同阶段复核再开下一轮，或有依据地 `complete` |

小头更新使用已标注的累计真机 replay；BC 示教缓存不冒充带 reward 的真机 transition。
每次 `train-ac` 都显式给更新预算，日志报告实际 Q/A 更新数、样本数和数据复用比例。
不是固定“先等 Q 收敛再训练 A”，也不是4局采完后偷偷边运动边更新。

## 现场只需一条命令

助手完成相应阶段验收以后，在前台终端运行：

```bash
ur5e-rlt round cube_rlt_20260910_01 --execute
```

它自动加载/预热固定的 JAX 和 PyTorch 服务；4局期间不变更模型。
无需另外启动服务，亦不需要独立 `infer-init`：每局提示 `r` 后回关节初始位、开爪；
你复位场景、Enter 开始；执行时 `q` 停止本局。停止控制后，再输入：

- `s`：任务成功；`f`：正常失败。人工结果仅在最后实际动作点赋稀疏终端奖励1/0。
- `t`：纯时间预算耗尽且未判正常失败，仅限达到 max_chunks；按截断保存，允许从复位前末态 bootstrap。
- `a`：中止，不进入 TD replay；不是失败。设备异常也按中止留存。

默认延续一次 close→open，释放后保持至少1秒；回合成功**由人判定**，不会把“发出open”自动当成功。
当前成功提示是“抓起、放下并完全释放”，是否增加固定落点区域须在正式 reference 采集前明确。
首次在线探索前还需查看 `.002` 的实际各关节/夹爪尺度；它是 donor 的 **tanh 前标准差，不是 .002 rad**。
本轮软件授权不等于已经启动这些现场回合。

需要暂停时，在局间输入 `q`；按 `status` 给出的轮次恢复：

```bash
ur5e-rlt round cube_rlt_20260910_01 --execute --resume ROUND_ID
```

恢复只补剩余局数。已完成但未标注的局先补标注，绝不重放；中断且没有完整末态的局记 aborted。
也可用 `label --episode PATH --result success --reason ...` 补标；已提交标签不可覆盖。
4局全部结束后必须复核再开新轮。`round --dry-run` 仅显示计划，不检查硬件、不加载模型、不生成真机样本。

## 日志与恢复

全部运行大文件在 Git 忽略的 `logs/rlt/<RUN>/`，本版尚未自动迁移到外置4TB数据盘：

| 文件 | 用途 |
| --- | --- |
| `run.json / state.json / invocations.jsonl` | 基线参数内容清单与归一化、不可变配置、当前阶段/选中版本、命令与实现代码摘要 |
| `cache/` | 原生 prefix/mask、规范 reference、量化前状态；以及冻结 token 后的 z 缓存 |
| `token_train.jsonl / token_diagnostics.*` | 每更新loss/梯度/资源；重建、token置零和跨样本打乱诊断 |
| `heads_train.jsonl / heads_diagnostics.*` | BC−Q、梯度、更新计数、全replay TD/Q与已知终端回报校准、物理关节/夹爪误差 |
| `rounds/<ROUND>/spec.json` | 本轮不可变 token/head 哈希、执行者、种子；配置由 run_id 固定 |
| `rounds/<ROUND>/episode_*/` | 逐chunk原始双图、状态、规范动作、请求/实际前缀、命令与测量轨迹、夹爪事件、复位前末态及人工标签 |
| `rounds/<ROUND>/summary.json` | 4次尝试的成功/失败/截断/中止/待标注与时长 |
| `telemetry/<INVOCATION>/` | 2秒资源采样、30秒摘要；整卡余量、主机与服务进程内存；采样失败单独报告 |
| `checkpoints/heads/` | 每个 learner 更新的 A/C、target Q、两个 optimizer、RNG、计数与 replay 来源摘要 |
| `DECISIONS.md / decisions.jsonl` | 选哪个 checkpoint、继续/切阶段/回退、原因和证据文件哈希 |

A/C 每更新保存并保留最近20个；被阶段决定或回合引用的版本长期保留，自动清理仅针对本 run 的未选旧小头。
Token 是大 transformer：包含 encoder/decoder/optimizer 的恢复点每100更新保存，不逐步复制。
中断恢复读取最后完整 checkpoint，不宣称能恢复尚未提交的梯度更新或凭空补齐丢失的真机末态。
token 500更新范围内的默认完整恢复点合计可能达数十GB，开训前看磁盘余量。

## 开发检查

```bash
python -m pytest -q
.venv/pi05/bin/python scripts/check_rlt_native.py
.venv/pi05/bin/python scripts/check_rlt_native.py --checkpoint joint5_sft_20260909_01:1000
```

前者使用开发/硬件环境；后两者是原生 dummy/已有 SFT 的固定噪声离线动作对比，不训练、不连接设备。
测试证据不替代各阶段正式训练和真机验收。
