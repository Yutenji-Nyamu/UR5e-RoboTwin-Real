# D435i 可选深度采集

更新：2026-10-06。`--depth` 已实现，两台 D435i 连续相机落盘实测通过。

## 使用

按原流程完成 `ur5e-collect-init` 后：

```bash
ur5e-collect block_drawer_close --note "1005" --depth
```

也支持 `ur5e-real collect ... --depth`。不传参数走原来的 `DualColorCamera`；
不修改全局 YAML 开关。键盘、结果复核和原 RGB 文件/CSV 约定不变，深度不进入 MP4。
`--depth` 成功启动会打印 `[DEPTH]`。相机失败会报错，不能静默降级为 RGB。

实现入口：[相机](../../../src/ur5e_real/hardware/rgbd.py)、
[写入器](../../../src/ur5e_real/collection/depth.py)、
[采集器](../../../src/ur5e_real/collection/session.py)。
每台设备一条 pipeline 同时采 color BGR8 与 depth Z16，按各自 RGB 几何网格对齐；
当前 lab 配置为 640×480@30 Hz，保存约 10 Hz。

## 文件与数值契约

```text
raw/camera/cam_dual_<run>/
  head/frame_00001.png             # 原 RGB
  wrist/frame_00001.png
  head_depth/frame_00001.png       # RGB-aligned Z16，单通道 uint16
  wrist_depth/frame_00001.png
  camera_calibration.json
  rgbd_frames.csv
```

- 深度无损 PNG，0 表示无有效测量；米值为 `uint16 * depth_scale_m_per_unit`。
  不做滤波、补洞、伪彩色替代或单位猜测。本机约为 0.001 m/unit，代码从设备读取。
- 标定包含每台设备 serial、固件、SDK、流规格、深度单位、RGB/native-depth/aligned-depth 内参，
  depth→color 外参。平移单位米，旋转数组列优先。它不包含相机到机器人/工具的手眼外参。
- `rgbd_frames.csv` 每组两行（head/wrist），记录 `frame_idx`、控制器时间、RGB/depth 文件名、
  两流各自帧号/时间戳/时间域、主机接收 wall/monotonic 时间、非零深度比例。
  原 `sync_action_cam` 四列表头不变；以 `frame_idx` 关联。
- raw schema 仍为 **3**，manifest 新增 `depth_recording.version=1`、enabled、encoding、alignment、
  units、invalid_value、complete_pairs；具体路径加在 `paths` 中。旧读者可继续只读取 RGB/q/TCP。
- 深度是几何配准结果，未另存原生 depth 网格；如后续需要重新配准，再显式增补原生流。

相机对齐不等于时间硬同步。采集循环先收 RTDE，再取两相机帧；保留不同时间源供后续对齐，
不把控制器时间解释为相机曝光时刻。两台相机也未接硬件同步线。

## 故障与资源处理

启动先完成两台相机预热、标定、完整 RGB-D 帧验证，再连接 RTDE/夹爪和进入 freedrive。
第二台启动失败会停止已启动的第一台。重复帧号、倒退/非有限时间、时间域切换、缺流、
错误 shape/dtype 或写文件失败会中止并清理资源；manifest 标 failed。
SDK 取帧等待上限每次 2 秒，第一版仍为同步循环，不承诺毫秒级停止响应。

双 RGB、双 depth 写成功后才发布完整索引。磁盘出错时可能残留该组部分 PNG，
不得仅数 PNG 判断完整性，应同时检查 manifest 状态/计数、原 sync 与旁表。
记录保留失败前的 RTDE 样本；不把失败 session 自动标成功。

## 实测与复现

```bash
conda activate RoboTwinSimReal
python examples/smoke/rgbd_recording.py \
  --config configs/lab.yaml --seconds 60 --output /tmp/rgbd-check
```

输出目录必须不存在；这是**生产相机类 + 生产深度写入器**，保存四张 PNG 并逐张回读深度。
不连接 RTDE、串口或 freedrive，旁表控制器时间为 NaN。证据见
[连续落盘 JSON](evidence/rgbd_recording_20261006.json)。

| 指标 | 2026-10-06 结果 |
| --- | --- |
| 相机 | 双 D435i；USB 3.2，固件 5.17.0.10 |
| 保存 | 60 秒，600 组 RGB-D，2,400 张 PNG |
| 深度回读 | 1,200 张 uint16，逐像素相等全部通过 |
| 读帧 + 四 PNG 写入 p50 / p95 / max | 60.55 / 65.04 / 106.84 ms |
| 保存起点间隔 p50 / p95 / max | 100.00 / 100.05 / 112.50 ms |
| 双深度平均增量 | 每组约 111 KB，600 组约 66.6 MB |
| 四路 PNG 总体积 | 680,650,688 bytes |

额外回读校验不计入“读写耗时”，但影响保存起点间隔。
这是 `/tmp` 文件系统上的一分钟组件实测，未涵盖数据盘长时吞吐、预览/视频写入或 RTDE 同时采集，
不等于现场完整示教验收。采集回归通过 mock 覆盖不加参数、事件 close/open/close、初始缺深度、
深度写失败与清理；SDK mock 覆盖双流、启动失败和重复帧。

首次调研另有 [60 组 RGB-D/标定探测](evidence/realsense_rgbd_20261006.json)，
设备序列号在公开证据中已去标识；实际采集标定文件仍保留原始设备信息。

官方 API 依据：[RGB-D 对齐](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)、
[投影与坐标](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。
