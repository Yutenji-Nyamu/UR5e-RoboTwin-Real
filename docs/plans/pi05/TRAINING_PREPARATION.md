# π0.5 开训准备与短测

2026-09-09。用户授权准备训练入口并测试几个 batch；每档只做几步。
本轮边界：batch=2/4/8，每档3个 optimizer step；再补batch=8/两个loader进程3步。
**共12个真实模型optimizer更新，不启动3000步正式拟合，不连接真机**。

## 已实现的训练工具

- batch 支持1..128，并拒绝大于样本总量的设置，避免原生 drop_last 产生空 loader。
- 默认关闭随机图像增强；可用 `--image-augmentation` 恢复。原生 loss、flow噪声、模型参数结构不变。
  本地仅继承原生模型，控制传给原生图像预处理的 train 标志；vendor源码锁未改变。
- 每2秒采样GPU显存已用/余量、利用率、功耗、温度，以及系统available内存、主/子进程RSS、swap。
  每30秒终端摘要；每步记录loss/LR/数据等待/同步计算耗时/JAX分配器统计，常规每10步打印训练指标。
  定时采样峰值与JAX分配器峰值分开记；不能把预留显存当作活跃tensor量。
- 每500步及最终保存，间隔可配。每500步及最终做固定训练观测的离线评估，间隔可配/关闭。
  中间评估直接使用当前参数，**不使用捕获旧参数的 Policy.module_jit，也不在GPU另载完整模型**。
  评估来自不可变LeRobot数据，排除 action_is_pad，独立统计实际q6/夹爪以及前6步。
- 完成保存的中间checkpoint发布“仅离线评估”的manifest；不放宽serve/execute审计门槛。
  最终仍须冻结、参数、optimizer重载核验；重载前先释放训练态以控制峰值。
- recipe v2把固定LR schedule与本次停止步数分开；同batch/增强/数据/optimizer recipe可以扩步续训到schedule上限。
  旧recipe v1不自动升级。扩步不是自动重置optimizer或改学习率。
- `--benchmark`只允许2..5步，不保存checkpoint、不跑评估；每档独立进程。

## 数据与权重

- 仍用 `ur5e/pick_place_cube_joint_5_v20260908`：5条、414个样本、10Hz、H=50。
- 复用本机已有的官方 `gs://openpi-assets/checkpoints/pi05_base/params` 缓存；源缓存不变。
  已放入原生缓存 `.venv/openpi-assets/openpi-assets/checkpoints/pi05_base/params`；
  `.venv/pi05-base-cache-20260909`保留为兼容符号链接，旧运行记录的路径仍有效。
  20个文件合计12,441,721,931字节。默认GCS参数路径已经验证直接命中本机缓存，不再重复下载。
  全部大小和CRC32C与官方GCS对象清单一致；不是任务微调权重或随机模型。
  完整校验记录：`outputs/pi05_base_cache_verification_20260909.json`（本地artifact）。

## 测试记录

- 完整单元/loopback回归：92项通过；ruff通过；188个隔离环境包兼容检查通过。
- `dummy_training_tools_20260909_01`：原生dummy模型3步；第2/3步保存与中间评估、
  最终参数/optimizer重载、原生Policy推理通过。此项不是完整基础模型性能证据。
- dummy后续从第3步续到第4步成功；native恢复参数/optimizer/step，LR继续到2.0362584e-5，未回到初值。
- 原生loader、RGB训推一致、delta/quantile/inverse、缺失左腕mask的真实414样本检查重新通过。

### 真实基础权重实测

每档3步；只用后2步粗测。单位GiB。完整机器记录见 [短测结果](SHORT_PROBE_RESULTS.json)。

| batch | workers | 同步计算s/步 | 等数据s/步 | 样本/秒 | JAX活跃分配峰值 | 整卡已用采样峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 0.502 | 0.168 | 2.99 | 12.81 | 16.94 |
| 4 | 0 | 0.847 | 0.300 | 3.49 | 13.33 | 16.94 |
| 8 | 0 | 1.572 | 0.545 | 3.78 | 14.39 | 16.95 |
| 8 | 2 | 1.539 | 0.040 | 5.07 | 14.39 | 16.95 |

- 吞吐按同步计算+取数耗时计算，不含首步JIT、日志、初始化、保存、评估和哈希审计；不是整段进程吞吐。
  workers预取和短窗口可能使长跑速度不同，不能据2个warm step作精确性能承诺。
- 这几档整卡可用显存最低仍约30.44GiB；系统available内存最低约86.15GiB。
  定时采样可能漏过瞬时峰值，另列JAX allocator峰值；二者不等同，整卡值也包含桌面/驱动。
- 所有真实档位冻结参数哈希不变、expert参数改变，loss/grad有限。batch2另外实际完成完整保存、
  参数与optimizer重载、当前训练态离线评估；其余3档是benchmark-only，不落大checkpoint。
- batch8的workers0/2在相同初始化、数据顺序及3步更新后，最终模型参数、optimizer哈希和loss/grad指标完全一致；
  这次加载并行度调整没有改变短测的训练结果。
- batch2第3步checkpoint位于 `checkpoints/pi05/pi05_ur5e_joint_action_expert/joint5_batch_probe_20260909_01_b2/3`，
  实际磁盘占用约8.8GiB；保存耗时约11.9秒。状态是development_smoke_only，不能用于execute。
- 独立 `evaluate` CLI实际重载该checkpoint通过（5个固定观测，207个有效目标，排除43个padding目标）。
  结果为 `outputs/pi05_base_smoke_checkpoint_eval_20260909.json`；该小样本检查不代表任务成功率。
- 结论：推荐正式首轮 `batch=8 / workers=2`。较2/0的样本吞吐粗测提高约70%；
  比8/0提高约34%。当前证据支持先减少数据等待，不必为占满48GB继续扩大参数搜索。
  这不证明收敛时间等比例缩短；仍按有效关节/夹爪误差判断拟合。

### 下一轮正式训练建议（本轮未启动）

保持expert-only、LR2.5e-5、warmup100、H50/10Hz、关闭图像增强、EMA关闭。
采用batch8/workers2，固定schedule=3000，先运行到1000步观察；每500步保存及评估。
之后可在相同recipe下扩展至3000步，保留optimizer；不要把本轮3步的开发checkpoint当作拟合完成。

## batch 与旧DP对照

batch是每次optimizer更新一起计算的训练样本数，不是轨迹数，也不是动作窗口H或执行步数K。
π0.5的一个样本是某时刻观测和未来50步动作；batch=8即8个这样的窗口共同产生一次更新。

已读取实际Hydra记录：

- DP `20260904_160554`：6条入集，其中5条训练、1条验证；batch=111、gradient_accumulate_every=1、600 epoch。
- DP `20260905_150221`：17条入集；batch=72、gradient_accumulate_every=1、600 epoch。
- 两次默认batch来自 `min(128, shortest_episode)`，是为保证小验证集仍能凑出一个batch。
  DP模型/窗口与π0.5不同，不能照抄111或72。

短测只用于判断大致吞吐和资源余量，不能从两次预热后更新推断收敛、成功率或稳定长跑性能。
