# 模块初始化与输入接口

Gitee 固定版本 `1392ab585874701485c32372828a873bd0ac7644`；本轮重新 fetch 后仍为最新提交。作者源文件保持原样，真实数据适配集中在 UR5e 分支。

| 模块 | 初始化来源 | 本轮核验 |
|---|---|---|
| Video expert | OpenWAM-Alpha Real Single Arm Franka 的 Wan 视频专家 | 825 个张量加载，兼容项完整；demo_embedding 按作者零初始化 |
| Action expert | 同一 Alpha 检查点 | 824 个张量加载；本轮学习单臂 joint7 输出接口 |
| Proprio 编码器 | 同一 Alpha 检查点 | 2 个张量加载 |
| Track expert | 作者 interpolate_from_video 初始化 | 203 项直接复制、354 项插值变形、23 项按作者初始化 |
| 空间/时间读取、角色、进度等新增模块 | 作者初始化实现 | 在本轮实测 RGB-D、角色、位移、进度与动作上联合学习 |
| 视频/Track VAE | Wan2.2 TI2V 5B 官方 VAE | 48 通道，冻结后离线编码 |
| 文本编码器 | 同套 Wan UMT5 与 tokenizer | 冻结，五条任务指令离线缓存 |

模型分组参数量：Video 4,999.8M，Track 688.6M，Action 1,021.0M，Focus 43.8M，Proprio 0.3M，总计约 6.754B。

Alpha 的机器人动作先验为 EEF10。本轮按照实测数据使用 6 个关节位置增量与绝对夹爪状态，嵌入 80 维动作的 10–16 槽；新增本体索引为 9。输入状态与动作分别使用训练集统计归一化，六关节单位为弧度，夹爪映射为 2g−1。

视频布局：head 320×192 在上、wrist 320×192 在下；共 9 帧，对应 3 个 VAE 时间位置。Track 只取固定 head，320×240 中心补到 320×256。RGB、实测深度与前景 mask 各编码为一帧条件。

完整加载报告：`runs/smoke_alpha_b2/initialization.json`，正式运行另存自己的同名文件。仓库与模型来源锁定在 `sources.lock.json`。

模型权重 SHA-256：

- Alpha：`76e53e963525d9eebdb70e5285908c6497082270de66f9227874844e96bc43bb`。
- SAM 3.1：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- RAFT：`ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`。

来源：[OpenWAM-Alpha](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka)、[Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)。
