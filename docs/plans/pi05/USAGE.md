# π0.5 joint 接入操作入口

本页列操作命令，实际跑过哪些检查以 [实施记录](IMPLEMENTATION.md) 为准。
模型运行在独立 Python3.11/JAX 环境；硬件环境保留现有 DP/PyTorch，二者只用 localhost WebSocket 通信。
`offline` 默认不连接硬件；`shadow` 只读；`prepare --execute`、`infer --mode execute` 才会运动。
当前适配不涉及 RoboTwin 仿真、charger 或 RLT。

## 1. 环境

从仓库根目录执行；已有锁定的 `.third_party/RoboTwin`。

```bash
# 无 uv 时，只在项目中安装引导工具，不改原有训练依赖。
python -m pip install --target .venv/bootstrap uv==0.8.22
bash scripts/bootstrap_robotwin.sh
bash scripts/setup_pi05.sh

# 在原硬件环境中，只补两个通信包；不安装 π0.5/JAX 到该环境。
conda activate RoboTwinSimReal
python -m pip install --no-deps websockets==15.0.1 msgpack==1.1.1
```

后续使用模块入口不要求更新旧环境中的 console script。
模型依赖锁在 `integrations/pi05/requirements.lock`，数据/预训练下载与编译缓存不进 Git。
native commit 不匹配或存在已托管兼容补丁之外的变动时拒绝加载，不自动重置用户改动。
当前三个小补丁分别修复示例配置导入、锁定Orbax的asset回调和下载错误传播；不修改模型、loss或推理算法。

## 2. 五条数据：审计 / 不可变导出

```bash
python -m ur5e_real.adapters.robotwin_pi05 data \
  --selection configs/pi05_cube_joint_5.json \
  --data-root /data/robotics/ur5e-real

.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 data \
  --selection configs/pi05_cube_joint_5.json \
  --data-root /data/robotics/ur5e-real \
  --lerobot-home /data/robotics/ur5e-real/pi05/lerobot
```

如果目标已存在，不重复执行导出；变更筛选/预处理时使用新的 `repo_id`，不覆盖原始记录或旧导出。
本版固定 v3 / actual_q / rad / 一次 close→open；旧 TCP 数据会拒绝。
10Hz 线性重采样 q/TCP、夹爪零阶保持、图像最近邻；0.2秒 source gap 明确记录，不当作均匀10Hz原始采样。
观测 q[t] 对应标签 q[t+1]；最后训练观测至少是 open 后1秒，额外0.1秒作末帧标签。

后续命令共同使用：

```bash
PI05_DATA=/data/robotics/ur5e-real/pi05/lerobot/ur5e/pick_place_cube_joint_5_v20260908
.venv/pi05/bin/python integrations/pi05/check_native.py --dataset "$PI05_DATA"
```

检查原生 LeRobot loader、训练/RPC 图像与 state/action 一致、缺失左腕 mask、quantile/delta/pad/inverse 往返。
导出保存原生统计、原始来源审计、动作契约和 parquet SHA-256；训练/服务校验数据版本和统计未变。

## 3. 原生 SFT

```bash
# 开发单步：batch1、warmup0；会加载官方完整基础权重，不是随机小模型。
.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 train \
  --dataset "$PI05_DATA" --exp-name joint5_smoke_01 \
  --steps 1 --batch-size 1 --warmup-steps 0

# 正式最小拟合（本轮不自动启动）：先1000步，再据离线结果决定。
.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 train \
  --dataset "$PI05_DATA" --exp-name joint5_sft_01 \
  --steps 1000 --batch-size 2 --warmup-steps 100
```

可用 `--params /absolute/path/to/pi05_base/params` 指定已下载权重。
首次下载基础参数约12.5GB，网络慢时不能把尚未开始的 backward 记为通过。
显存不足改 batch1，并换实验名；不自动切 full finetune。EMA关闭，W&B关闭，单GPU，workers0。
只训练 action expert 和 action/time 投影；前后逐参数组哈希确认 backbone 不变、expert 改变；
保存 native params/optimizer/norm 并实际重载核验。未通过检查的 checkpoint 不获得本地服务就绪标记。
同一实验不覆盖；`--resume` 只接受完全相同 recipe 的中断续训，增加预算用新实验从参数继续并重建optimizer。
中间checkpoint用于续训；只有最终通过冻结/参数/optimizer重载核验的checkpoint能被服务加载。
续训恢复参数、optimizer和step，数据shuffle迭代器会重建，不承诺逐样本逐位复现。

不下载完整权重的开发测试入口：`integrations/pi05/smoke_native.py --dataset ...`，需模型Python运行。
它使用原生dummy Gemma随机参数，验证编排而非完整π0.5拟合；全尺寸架构只做抽象参数计数，
不可把该测试显存/时延写成真实base模型指标。

## 4. 服务与离线

```bash
PI05_CKPT=checkpoints/pi05/pi05_ur5e_joint_action_expert/joint5_sft_01/1000
.venv/pi05/bin/python -m ur5e_real.adapters.robotwin_pi05 serve \
  --dataset "$PI05_DATA" --checkpoint "$PI05_CKPT" --port 8005
```

等两次 warmup 和 `[READY]` 后，在硬件环境的另一个终端中：

```bash
python -m ur5e_real.adapters.robotwin_pi05 infer \
  --dataset "$PI05_DATA" --mode offline --port 8005 \
  --run-id 20260908_153254 --index 30
```

报告真实端到端时延、关节误差；H=50 输出绝对 q，第13列是夹爪。**不手动累加 delta**。
服务绑定127.0.0.1；客户端校验同一完整动作/数据契约，有界超时后断开，绝不复用迟到响应。
开发 smoke checkpoint 可用于离线/shadow，不能用于 execute。

## 5. 现场阶段：需人在场，不属于离线开发验证

```bash
# 先只打印目标，不连接硬件。
python -m ur5e_real.adapters.robotwin_pi05 prepare \
  --lab-config configs/lab.yaml --contract "$PI05_DATA/ur5e_adapter/contract.json"

# 本人确认整臂直达路径无障碍、Remote Control / 停止状态后，才加 --execute。
# 低速关节 home，不沿用旧 TCP moveL home。
python -m ur5e_real.adapters.robotwin_pi05 prepare \
  --lab-config configs/lab.yaml --contract "$PI05_DATA/ur5e_adapter/contract.json" --execute

python -m ur5e_real.adapters.robotwin_pi05 infer \
  --dataset "$PI05_DATA" --lab-config configs/lab.yaml --mode shadow --chunks 10

# 受控真机测试时才执行：开始会开爪；Ctrl-C/异常会停止 joint 控制。
python -m ur5e_real.adapters.robotwin_pi05 infer \
  --dataset "$PI05_DATA" --lab-config configs/lab.yaml --mode execute --chunks 30
```

shadow 不创建串口、运动脚本或 servo 输入 recipe；夹爪状态由 `--shadow-gripper 0/1` 指定，非实测开口。
execute 要求当前 q 对齐 home（各轴0.03rad内），不偷偷自动回 home；TCP offset 必须与采集一致。
先停机检查，再将持久寄存器 prime 为 actual_q/mode0；新脚本需正确 program ID＋本次nonce＋runtime2回执。
K=6，关节500Hz插值，上限0.6rad/s；允许最大2倍时间拉伸，完整目标预检；偏差过大时不执行对应夹爪事件。
短调度延迟不追赶式突发发点；命令生产超时、状态/相机过旧、超demo范围等均终止控制。
一次close→open后保留1秒稳定hold再停止；不引入连续遥操作。

这些是软件约束，不是整臂碰撞规划或现场安全认证。新 joint URScript/watchdog 与运动时序仍须现场核验；
旧 TCP DP/手工回放继续独立，不把它们已有的真机结果算作 π0.5 成功。
