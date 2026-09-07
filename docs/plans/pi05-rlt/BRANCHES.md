# rlinf_fastwam 分支快照

采集日期：2026-09-07。仓库：[Yutenji-Nyamu/rlinf_fastwam](https://github.com/Yutenji-Nyamu/rlinf_fastwam)。共 37 个分支；`origin/HEAD` 符号别名不另计。

本表固定的是调研时的 branch tip，不代表完整历史已审查或训练结果已复现。提交日期按仓库记录，分支名中的算法名也不能单独证明当前文件实现。建议阅读顺序见 [调研记录](RESEARCH.md)；采用方案见 [主规划](README.md)。

## 上游基线（1）

| 分支 | 固定 commit | 提交日期 | 阅读定位 |
| --- | --- | --- | --- |
| `main` | [`8138d6700e3838250c1139289ebfba43d48ff7de`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/8138d6700e3838250c1139289ebfba43d48ff7de) | 2026-07-19 | 上游基线 |

## RLT 家族（7）

| 分支 | 固定 commit | 提交日期 | 阅读定位 |
| --- | --- | --- | --- |
| `codex/rlt-dvac-pure-reference-bc` | [`f0aaf4b71669fad38d11ac85c90670386242c29d`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/f0aaf4b71669fad38d11ac85c90670386242c29d) | 2026-08-29 | RLT 变体；关键差异抽查 |
| `codex/rlt-dvac-success-episode-bc` | [`848b61278687702ea717c56b3734f1486cea3b95`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/848b61278687702ea717c56b3734f1486cea3b95) | 2026-08-26 | RLT 变体；关键差异抽查 |
| `codex/rlt-pi0-robotwin` | [`2b8199d8ab2e7b110994fd3234bf7007196c3af9`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/2b8199d8ab2e7b110994fd3234bf7007196c3af9) | 2026-07-31 | 早期 RLT；重点源码/报告 |
| `codex/rlt-teacher-dvac-weighting` | [`74c715515c94fd367aff274871bf9488e95ff6b3`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/74c715515c94fd367aff274871bf9488e95ff6b3) | 2026-08-25 | RLT 变体；关键差异抽查 |
| `codex/sz-rlt-checkpoint-diagnosis` | [`f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4) | 2026-08-23 | RLT 契约/续训诊断；重点差异 |
| `codex/sz-rlt-dvac-pure-single-gpu` | [`30349428c37a008b95342121c1455debfeb4805e`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/30349428c37a008b95342121c1455debfeb4805e) | 2026-09-03 | RLT 变体；关键差异抽查 |
| `codex/sz-rlt-pi0-robotwin-ar` | [`d3acd650869d376c14c1406d90865ef90d680432`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/d3acd650869d376c14c1406d90865ef90d680432) | 2026-09-05 | AR RLT；首选 donor；重点源码 |

## π0.5 接入/实验（7）

| 分支 | 固定 commit | 提交日期 | 阅读定位 |
| --- | --- | --- | --- |
| `codex/sz-pi05-grpo-dvac-adv-chunk-positive` | [`9d01a7208c563d72d1b86e753fb1c103f95e6905`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/9d01a7208c563d72d1b86e753fb1c103f95e6905) | 2026-09-07 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-pi05-grpo-dvac-st-positive` | [`5a5ec53dc39de9dadb3cecf5b9f36a678de47a4a`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/5a5ec53dc39de9dadb3cecf5b9f36a678de47a4a) | 2026-09-07 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-pi05-online-bc` | [`1d70a698b81fcd730ac614d5d88db3e8f20c2af7`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/1d70a698b81fcd730ac614d5d88db3e8f20c2af7) | 2026-09-07 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-pi05-online-bc-dvac` | [`e2c8656e078e166ac67dc90aaa8e965a7b8b7ffe`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/e2c8656e078e166ac67dc90aaa8e965a7b8b7ffe) | 2026-09-07 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-pi05-robotwin-rl` | [`ae7e5da72acf4a47a54cecff4f802ebc174b397a`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/ae7e5da72acf4a47a54cecff4f802ebc174b397a) | 2026-09-03 | π0.5 接入；重点配置 |
| `codex/sz-sidney-pi05-current-rlinf` | [`1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591) | 2026-09-06 | π0.5 转换/parity；重点源码 |
| `codex/sz-sidney-pi05-grpo-dvac-adv` | [`746f396377de78c9900b6792fd9815530d1bd6dd`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/746f396377de78c9900b6792fd9815530d1bd6dd) | 2026-09-07 | tip/目录/相关路径盘点；非本轮核心 |

## π0 / DVAC / 其他实验（19）

| 分支 | 固定 commit | 提交日期 | 阅读定位 |
| --- | --- | --- | --- |
| `codex/dsrl-pi0-robotwin` | [`48a775db09c16c455aeba7b0600c920e7c80d534`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/48a775db09c16c455aeba7b0600c920e7c80d534) | 2026-07-29 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/idea2-dvac-pi0-robotwin` | [`61996e15cc7f5a32bd6012b61b20893d94636c82`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/61996e15cc7f5a32bd6012b61b20893d94636c82) | 2026-08-20 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/idea2-dvac-residual-downweight` | [`afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8) | 2026-08-23 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/idea2-dvac-train-weighting` | [`145fa810f1d8baee23012922b81e496661d61cf5`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/145fa810f1d8baee23012922b81e496661d61cf5) | 2026-08-21 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/ogpo-pi0-robotwin` | [`5d5c84e3ac4efa1713a4139a05ac1b776e634ed3`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/5d5c84e3ac4efa1713a4139a05ac1b776e634ed3) | 2026-08-07 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/qam-pi0-robotwin` | [`ff8e28ef6a4d485642e695b6b76c5c84187e134e`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/ff8e28ef6a4d485642e695b6b76c5c84187e134e) | 2026-08-01 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-7d07a421-grpo-pi0-robotwin` | [`02b9488329bf0e141a4dc3527e5885bfeae87088`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/02b9488329bf0e141a4dc3527e5885bfeae87088) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-current-dsrl-pi0-robotwin` | [`4ec52bd8d3dad1744b3f61c76636314b9d69c7a6`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/4ec52bd8d3dad1744b3f61c76636314b9d69c7a6) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-current-pi0-dvac-grpo` | [`66c863bc5a45e90cb5161b30af54355b1104c810`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/66c863bc5a45e90cb5161b30af54355b1104c810) | 2026-08-24 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-current-pi0-dvac-grpo-w0to5` | [`ab0988498a04311385600210c5b5ad34f16df1ce`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/ab0988498a04311385600210c5b5ad34f16df1ce) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-current-pi0-dvac-observe` | [`800baf80d6eab64169cf0e691eb04a681a093ee9`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/800baf80d6eab64169cf0e691eb04a681a093ee9) | 2026-08-22 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-grpo-dvac-action-adv` | [`9fc8b403d9158f19d3d5c15065453e275e803e87`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/9fc8b403d9158f19d3d5c15065453e275e803e87) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-grpo-dvac-action-adv-fix` | [`7006ad20a6ad7357c48e6d20e3fcabed6c5609e4`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/7006ad20a6ad7357c48e6d20e3fcabed6c5609e4) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-pi0-online-bc` | [`2467d997831166b70444b0c99d5198a2d3dfc8f6`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/2467d997831166b70444b0c99d5198a2d3dfc8f6) | 2026-09-05 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-pi0-online-bc-dvac` | [`912808c74c8df218e4a58bab850b18a8b9215b8a`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/912808c74c8df218e4a58bab850b18a8b9215b8a) | 2026-09-05 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-ppo-dvac-action-adv-fix` | [`036e6bca07a84ff7cfe0c0578f7a81b6afc7355e`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/036e6bca07a84ff7cfe0c0578f7a81b6afc7355e) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-ppo-pi0-robotwin` | [`c41b7ff0008574799636abb37783ffbf16c6be11`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/c41b7ff0008574799636abb37783ffbf16c6be11) | 2026-09-05 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-prism-dvac-rank-rloo` | [`76723092146e17fbb32094267f68cb2de8979828`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/76723092146e17fbb32094267f68cb2de8979828) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-st-dvac-local-shard` | [`51dbeaebe94cb218908b388193315a69a9a6c99a`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/51dbeaebe94cb218908b388193315a69a9a6c99a) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |

## FastWAM（2）

| 分支 | 固定 commit | 提交日期 | 阅读定位 |
| --- | --- | --- | --- |
| `codex/sz-fastwam-action-dvac-adv` | [`b816d6d32355977d6c42d6dee9ed732e2ab317ae`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/b816d6d32355977d6c42d6dee9ed732e2ab317ae) | 2026-09-03 | tip/目录/相关路径盘点；非本轮核心 |
| `codex/sz-fastwam-current-rlinf-grpo` | [`0d5daf6fa98de14865a74be143f51ef2972dab49`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/0d5daf6fa98de14865a74be143f51ef2972dab49) | 2026-09-05 | tip/目录/相关路径盘点；非本轮核心 |

## 归档（1）

| 分支 | 固定 commit | 提交日期 | 阅读定位 |
| --- | --- | --- | --- |
| `codex/sz-experiment-archive-20260906` | [`da19c0e16d55cb43f8cfebf637cae4f230223f64`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/da19c0e16d55cb43f8cfebf637cae4f230223f64) | 2026-09-06 | tip/目录/相关路径盘点；非本轮核心 |

## 使用边界

- AR RLT 与 Sidney π0.5 接入分别取代码与校验方法；不要把前者的 π0 token 权重直接挂到后者的 π0.5 权重上。
- 不把不同分支中的 norm stats、action transform、token layout、optimizer/replay checkpoint 混用。
- 移植代码保留原始许可/版权说明，并在后续实现记录中记清 donor commit 和本地补丁。
- 本轮没有 checkout/merge 这些分支到当前 UR5e 工作树，也没有运行其安装、训练或硬件脚本。
