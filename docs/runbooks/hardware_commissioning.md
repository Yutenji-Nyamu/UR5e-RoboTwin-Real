# Hardware commissioning

[简体中文](../zh-CN/runbooks/hardware_commissioning.md)

Use this order to isolate failures from the bottom up. Steps through section 3
are read-only and cannot move the robot or gripper.

## Current B81L baseline

Checked on 2026-09-01:

- workstation and UR5e share one static subnet; actual addresses are in local
  `configs/lab.yaml`;
- Dashboard `29999`, URScript `30001`, and RTDE `30004` are reachable;
- PolyScope `5.13.0`, robot mode `RUNNING`, safety `NORMAL`, Local control, no
  program running;
- gripper: CH340 adapter at stable `/dev/serial/by-id/...` path;
- configured head and wrist D435i devices both stream `640x480` color frames.

## 1. Power and PolyScope

1. Release the physical emergency stop and make sure the workcell is clear.
2. Press the teach pendant power button and wait for PolyScope.
3. On the initialization screen press **ON**, then **START**. The status should
   become `RUNNING` and safety should remain `NORMAL`.
4. Verify Ethernet is connected and the robot address matches `configs/lab.yaml`.
5. Do not load or start a motion program for the read-only checks below.

Remote Control must first be enabled in PolyScope settings. Use Remote mode for
external URScript/freedrive/manual replay. Use the validated Local-mode URP for
the RTDE servoJ path. Mode changes are human actions; automation must not switch
them silently.

## 2. Read-only robot checks

```bash
python examples/smoke/polyscope_status.py --config configs/lab.yaml
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10
```

Expected: Dashboard reports `RUNNING`/`NORMAL`; RTDE prints a stable 10 Hz TCP
pose. No PolyScope program is required for these reads.

## 3. USB checks

```bash
python examples/smoke/realsense.py --config configs/lab.yaml --frames 3
ur5e-real doctor --config configs/lab.yaml --hardware
```

If a RealSense serial is absent, reseat both ends of its data cable and prefer a
rear USB 3.x port. The gripper has no implemented identity/readback command, so
the doctor proves only that the expected serial adapter exists.

The first D435i color frames can be dark and strongly color-cast while automatic
exposure and white balance converge. The upstream
[librealsense OpenCV example](https://github.com/realsenseai/librealsense/blob/master/doc/stepbystep/getting_started_with_openCV.md)
also discards startup frames. This lab discards 60 frames (about two seconds at
30 FPS), because its measured color balance converged later than exposure;
adjust `warmup_frames` only after measuring the actual lighting.

## 4. First output checks

These require a human at the workcell and an explicit `--execute`:

1. Park the arm so the gripper is clear; issue one gripper `open`, then one
   `close` after visual confirmation.
2. Inspect the home target with `ur5e-real prepare --config configs/lab.yaml`,
   then add `--execute` after an on-site check to validate the slow home move and
   gripper opening.
3. Support the arm and test freedrive start/stop.
4. Record a short path; dry-run its replay; execute one bounded segment.
5. Separately load the servoJ URP and validate hold, millimetre step, ramp, and
   watchdog stop.

Never combine the first gripper, freedrive, replay, and servoJ tests into one
command. Each device must have an independently understood failure mode.
