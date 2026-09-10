# 新机器部署与自底向上检修

[English](../../runbooks/new_machine.md)

这是“从哪开始、怎样逐层验收”的总入口；[环境部署](setup.md)负责软件安装，
[前置条件](../PREREQUISITES.md)列依赖，[硬件调试](hardware_commissioning.md)解释设备故障，
[日常操作](operator_workflows.md)面向已验收现场。

**可复现边界：仓库支持重建代码/依赖和逐层调试，但 clone 后不能直接复现现有权重的真机效果。**
数据、模型、现场配置不在 Git；旧现场的工具、相机安装和原位也不会随代码迁移。
本手册于2026-09-10按源码、现有实机记录和旧项目文档核对，尚未在空白新电脑上完整重装验收。

## 0. 先确定迁移类型与设备清单

| 场景 | 必须带走/重做 |
| --- | --- |
| 重装当前电脑或换电脑，保留同一工位 | 备份数据和模型、记录 PolyScope 安装参数；重建环境、填写本机设备标识、逐层回归 |
| 新 UR5e / 新工具 / 相机位置或场景改变 | 重建安装与标定，确定新 home；先补采本机 joint 示教、导出新数据版本，再 SFT；不默认复用旧权重/范围 |
| 只检修一个设备 | 从第3节对应层开始；该层通过后再验上层，不先重装全部 Python 包 |

现有软件对应 UR5e e-Series、头/腕各一台 D435i、USB3数据线、CH340串口转接器和
当前串口夹爪、有线以太网、NVIDIA GPU。B81L“开5个断路器”仅是旧现场笔记，不是新工位接线规范。

在不提交的现场交接记录中填写：机器人/控制箱型号与序号、PolyScope版本、IP/网卡子网；
工具质量/重心/TCP offset、底座与工作台安装、限位和急停；夹爪确切型号、供电额定值/接线图、
USB转接器；相机串号→角色、安装位置/方向/照片、线缆、光照；数据盘挂载点与容量。
**旧文档未提供夹爪完整采购型号/电气针脚图，也没有可直接复用的新工位机械标定包**，
必须由设备手册和现场测量补齐；不能把 CH340 型号当夹爪型号。

## 1. 系统与依赖

1. 新装优先 Ubuntu 24.04 LTS；当前 Ubuntu25.04 是既有成功现场，不是推荐重装模板。
   先安装 Git、curl、CA证书、USB工具和编译工具：

   ```bash
   sudo apt update
   sudo apt install git curl ca-certificates gnupg usbutils build-essential
   ```

2. 按 [NVIDIA Ubuntu指南](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html)
   为实际 GPU/内核安装驱动并重启，`nvidia-smi`必须正常。不要照抄旧驱动版本强制降级，
   也不要把显示的“CUDA Version”当作已安装的 Python CUDA 包。
3. 按 [Conda Linux安装说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
   安装并初始化 bash，重新开终端确认 `conda --version`。
4. 按 [RealSense官方源配置](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md)
   添加与发行版匹配的 apt 源，然后安装 `librealsense2-dkms`、`librealsense2-utils`、
   `librealsense2-dev`和udev规则。当前上游列出20/22/24/26 LTS；内核/DKMS仍须匹配，
   不把旧项目的 `noble` 源硬写给任意 Ubuntu。已有发行版包时先核对版本，不叠装旧驱动。
5. 串口权限：`sudo usermod -aG dialout "$USER"`，注销登录后用 `id -nG`确认；
   再插拔串口/相机。内核已识别 CH340 时不编译旧 `CH341SER_LINUX` 副本，不用全局 chmod 777。

仓库与Python环境按[环境部署第1节](setup.md)安装。
`environment.yml`包括硬件/DP以及π0.5客户端的`websockets`、`msgpack`；
它固定关键依赖，但不是完整系统/Conda逐字节镜像，新机应记录最终解析包与驱动/内核版本。
π0.5模型在独立 `.venv/pi05`，不能把JAX依赖装进硬件Conda环境。

## 2. 本机配置、存储与短命令

复制 `configs/lab.example.yaml`为`configs/lab.yaml`并逐项填写，**不要覆盖已有配置**：

| 配置 | 确认方法 |
| --- | --- |
| `robot.host` | PolyScope上看到的实际地址；电脑有线网口使用同网段、不同地址，无IP冲突 |
| `robot.home_tcp_pose` | 先保持 null；第3节读取经现场确认的原位后填入6项，米/旋转向量弧度；不是欧拉角/角度制 |
| `gripper.port` | `ls -l /dev/serial/by-id/`，实际稳定路径；9600波特率及协议只适用于当前夹爪 |
| 相机串号/尺寸/帧率 | 枚举后逐台遮挡确认head/wrist；当前640×480、30FPS、保存10Hz、预热60帧 |
| `collection.data_root` | 数据目录实际存在、当前用户可写；后续raw/数据版本不要混到仓库源码中 |
| `servoj`路径 | 保留示例中相对配置文件的XML/URScript路径；不指向旧项目副本 |

存储见[存储手册](../STORAGE.md)。换电脑不要复制旧 `/dev/sda2` 或UUID；
Linux专用新盘可以选择本机文件系统，不要求NTFS。不在带数据的盘上执行格式化。
只读检查实际挂载：`findmnt -T /data/robotics/ur5e-real`、`df -h /data/robotics/ur5e-real`；
不要因为目录存在就把未挂载的数据盘当正常。

```bash
conda activate RoboTwinSimReal
scripts/install_operator_commands.sh
command -v ur5e-collect-init ur5e-pi05-infer
readlink -f /usr/local/bin/ur5e-pi05-infer
ur5e-real doctor --config configs/lab.yaml
```

安装脚本需要sudo写`/usr/local/bin`；它只建立指向**当前Conda环境入口**的软链接。
入口由`pyproject.toml`生成，代码来自editable仓库。安装时确认激活的是`RoboTwinSimReal`；
换路径/重建环境后重跑脚本，不复制旧机器的软链接或Conda目录。
此后短命令可在任意目录使用；下文`python examples/...`仍从仓库根目录、硬件环境运行。

## 3. 逐设备检查：先读，再单独输出

开机/解除制动需现场确认。PolyScope执行ON/START后应为RUNNING/NORMAL；
外部运动用Remote Control，无其他运动程序运行。有安全设置的控制器需允许RTDE服务。
软件只读检查不需要先播放URP。[端口与程序说明](../RTDE_STACK.md)

| 层 | 检查与通过标准 |
| --- | --- |
| 网卡/UR | `ip -br addr`、`ip route`；再运行`python examples/smoke/polyscope_status.py --config configs/lab.yaml`，确认状态/控制模式 |
| RTDE协议与TCP | `ur5e-real doctor --config configs/lab.yaml --hardware`；RTDE项须显示协议/控制器版本，而不只是端口通 |
| 同包joint/TCP | `python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10 --joints`；timestamp增长，q/qd/TCP/offset有限且单位正确 |
| 串口 | `lsusb`、`ls -l /dev/serial/by-id/`、`id -nG`；doctor只证明串口路径存在，不证明夹爪响应 |
| 单/双相机 | `realsense-viewer`逐台查看后**关闭viewer**；`python examples/smoke/realsense.py --config configs/lab.yaml --frames 3`须得到两幅640×480彩色帧 |
| 角色/光照 | 分别遮挡镜头确认head/wrist，预热后颜色正常；USB3连接稳定，无另一采集程序占用设备 |

如果某一项失败，先看[硬件排错](hardware_commissioning.md)，不要继续策略执行。

下面每条会实际改变硬件状态，逐条运行、观察，不整段粘贴执行：

```bash
# 机械臂停稳，夹爪周围净空；分别验证开/关，确认协议与供电
python examples/smoke/gripper.py open --config configs/lab.yaml --execute
python examples/smoke/gripper.py close --config configs/lab.yaml --execute

# 扶稳机械臂，分别验证自由拖动与退出；freedrive中不发move命令
python examples/smoke/freedrive.py start --config configs/lab.yaml --execute
python examples/smoke/freedrive.py stop --config configs/lab.yaml --execute
```

在PolyScope校准工具/TCP/负载，用已确认姿态的RTDE输出填写home。
`ur5e-real prepare --config configs/lab.yaml`只打印；确认回程路径后才加`--execute`。
若单独测URScript moveL，可用`examples/smoke/socket_move.py --help`，目标必须来自本机测量，
不要抄另一个工位的六个数。夹爪完整开/关由目视确认，当前没有开口/力反馈。

## 4. 集成验收：采集 → 重播 → 策略

1. `ur5e-collect-init`会**回TCP原位并开爪**；完成后恢复场景。
2. `ur5e-collect pick_place_cube --preview --note "new cell commissioning"`，先短录一条，
   `c/o/q`操作，结束后`s/f/a`标注。查`ur5e-real sessions --config configs/lab.yaml`；
   CSV须有`actual_q_0..5`/`actual_qd_0..5`，manifest为v3，有双图与夹爪事件。
3. `ur5e-replay RUN_ID --max-segments 1`先预览；准备原位与场景后才加`--execute`。
   socket是手动重播默认。独立500Hz TCP链可用`ur5e-replay RUN_ID --backend rtde --chunks 1`
   预览，再单独执行；这仍不是π0.5的joint策略验证。
4. DP路径按[训练](train.md)/[推理](infer.md)；π0.5路径按下一节及[π0.5操作](../../plans/pi05/USAGE.md)。
   新现场先选择干净5条joint示教，尾部至少保留最后open后1秒；不能把旧TCP-only数据改名当joint。

RTDE执行需确认真正的prime/运行握手，不只看打印chunk。程序、采集、重播、策略不得同时争用机械臂。

## 5. π0.5环境与资产迁移

```bash
python -m pip install --target .venv/bootstrap uv==0.8.22
bash scripts/bootstrap_robotwin.sh
bash scripts/setup_pi05.sh
.venv/pi05/bin/python -c 'import jax; print(jax.devices())'
```

须见GPU设备。上游由`robotwin.lock`锁定；兼容补丁由本地适配验证，不手改/重置上游模型目录。

| Git已有 | 需要另行准备，不能只复制一份params |
| --- | --- |
| 代码、测试、锁文件、选择清单、训练/推理说明 | 完整数据集`data/`、`meta/`、`ur5e_adapter/`和原始双相机图片 |
| 训练/执行摘要 | 完整`1000/` checkpoint，包括`params`、assets/norm、contract、verification及续训所需状态 |
| lab示例 | 实际lab配置、机器人安装/工具标定、相机固定位置 |
| 下载逻辑 | 重训时约12.5GB的`pi05_base/params`、tokenizer等缓存或可用下载网络；不能假设全离线自动启动 |

本次实验数据ID为`ur5e/pick_place_cube_joint_5_v20260908`，checkpoint目录见
[成功记录](../../plans/pi05/PHYSICAL_RESULT_20260910.md)。短命令按本机data_root定位
`pi05/lerobot/<dataset_id>`，模型在仓库`checkpoints/pi05/.../1000`；更换实验时init和infer均传同一`run:step`。

**当前移机限制：**导出的`ur5e_adapter/*.npz`仍引用原始图片的绝对路径，服务预热也读取这些图。
只搬LeRobot parquet或checkpoint会缺图。恢复相同工位时优先保持原`/data/robotics/ur5e-real`
布局，搬齐对应raw相机目录；不同根目录的自动重定位尚未实现，不能手改norm/contract绕过校验。
迁移后先跑`integrations/pi05/check_native.py --dataset ...`及操作手册的离线加载，核验资源哈希。

同一工位且迁移检查通过，再按序：

```bash
ur5e-pi05-infer-init 20260909_01:1000 --dry-run
ur5e-pi05-infer 20260909_01:1000 --dry-run
ur5e-pi05-infer 20260909_01:1000 --shadow --chunks 1
```

前两条不连硬件/不加载模型；shadow读取硬件但不输出动作。
之后现场确认再去掉init的`--dry-run`并运行infer的`--execute`。
如果home/TCP offset或数据契约不匹配，应重新标定/采集，不扩大限制强行通过。

## 6. 检修速查与交接记录

| 现象 | 先检查，避免误修 |
| --- | --- |
| `Unable to negotiate protocol version`偶发 | 已补3次fresh连接握手重试、间隔0.5秒；旧日志是1秒无回复，不足以证明协议不兼容。持续失败查网线/控制器服务，见[专题](hardware_commissioning.md#5-rtde启动握手偶发失败) |
| 端口通但无状态 | recipe字段、RTDE服务/控制器版本；doctor握手正常仍不证明500Hz流或输入寄存器可用 |
| `input ... in use` / prime失败 | 另一程序或Fieldbus占寄存器；先定位所有者，不当作握手超时无限重试 |
| 有chunk但不动 | Remote、机器人端程序身份/runtime/nonce、watchdog、反馈；见[RTDE链路](../RTDE_STACK.md) |
| 串口存在但夹爪不动 | 电源、协议/波特率、接线、设备占用和组权限；不是相机/模型问题 |
| 相机缺失/超时/偏色 | 线缆/USB3带宽、viewer占用、串号角色、udev、预热/光照 |
| 命令不存在/仍用旧代码 | `command -v`、软链接、Conda环境、editable仓库；在正确环境重跑安装脚本 |
| 数据根不可写 | `findmnt`和磁盘状态优先于chmod；不要对挂载中的数据盘直接运行修复 |
| 模型缺图/契约不符 | 完整资产、raw路径和版本；不能通过关掉校验来“修复” |

每次验收记录：日期、Git SHA、设备/驱动版本、实际配置、逐设备通过/失败、示教run ID、
数据/模型哈希、离线结果、现场操作者判断和失败日志。模型日志`completed`不等于任务success。

旧项目覆盖核对：`README.md`/`README_dev.md`中的网络、freedrive、夹爪、RealSense安装/单/双图、
RTDE读写、home、10Hz联合采集、HDF5/训练/推理，均映射到上面步骤与链接；
`Servoj_RTDE_UR5/README.md`的Local/URP流程被当前自动URScript替代；
`CH341SER_LINUX/README.md`的手编译驱动不再作为已识别设备的安装前置。
独立TCP采样/相机检查仍可单独运行，持久数据统一用collector保存。
旧ACT历史见[迁移记录](../MIGRATION.md)，不冒充已验收DP/π0.5链；
旧文档“转换后可删raw”的建议不沿用，原始数据保留。
