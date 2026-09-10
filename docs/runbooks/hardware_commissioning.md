# Hardware commissioning

[简体中文](../zh-CN/runbooks/hardware_commissioning.md)

Use this order to isolate failures from the bottom up. The software checks in
sections 2–3 are read-only; pendant power/brake release in section 1 is an on-site
operation. For installation, migration and acceptance across all layers, start
with the [new-machine checklist](new_machine.md).

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

Enable and keep Remote Control in PolyScope settings. External URScript,
freedrive, manual replay, and DP execution all use Remote mode. `ur5e-infer`
starts the robot-side servoJ program automatically while action targets remain
on RTDE.

## 2. Read-only robot checks

```bash
python examples/smoke/polyscope_status.py --config configs/lab.yaml
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10 --joints
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
5. Run the default RTDE servoJ policy from the inference guide; socket remains
   available only for comparison.

Never combine the first gripper, freedrive, replay, and servoJ tests into one
command. Each device must have an independently understood failure mode.

## 5. Intermittent RTDE startup negotiation

The reported `ur5e-collect-init` failure was at `RtdeTcpClient.connect()`, before
`move_linear()` or gripper opening. The home plan is printed before connection,
so the preceding `moveL` text is not evidence that a motion command was sent.

The installed `UrRtde==2.7.12` uses a 1 s receive timeout. After
`no data received in last 1 seconds`, its protocol call returns no answer and
raises `Unable to negotiate protocol version`, the same error used for refusal.
The evidence establishes a **startup reply timeout**, not a persistent protocol
incompatibility. The exact controller/network timing cause has not been captured.
UR distinguishes protocol request/acceptance from the subsequent recipe/start
exchange. [Official protocol guide](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html)

The 2026-09-10 fix in [the RTDE client](../../src/ur5e_real/hardware/rtde.py):

- Retries only connect/protocol/controller-version startup, at most 3 attempts
  with 0.5 s spacing. Each attempt uses a fresh object and closes failed sockets.
- Rejects a missing controller-version response. The old library keeps a failed
  socket attached and may skip negotiation if the same object is reused;
  the TCP servoJ startup now shares the fresh-connection helper as well.
- Cleans failed read-client recipe/start setup; does not retry recipe ownership,
  register writes, URScript submission, home commands or active motion.
- `doctor --hardware` performs a complete read-only protocol/version handshake
  then disconnects, instead of briefly opening/abandoning a bare port 30004 socket.
  The earlier probe's involvement is plausible, not a proven root cause.
- Keeps vendor stream timeouts and servoJ watchdogs unchanged. Exhausted startup
  becomes a clear `[BLOCKED] RTDE startup failed ...` in init, rather than an
  uncaught vendor traceback. A successful handshake is not a 500 Hz control test.

Validation used fault-injection/unit regressions without connecting hardware;
the intermittent field failure has not been physically re-induced. If retries
still exhaust, run the bounded read-only RTDE smoke check and retain the exact
log/controller state; inspect wired link/IP conflict, controller load/service
availability, and competing clients. A true unsupported protocol/recipe will
still fail. Do not automatically downgrade the protocol, disable guards or keep
re-running motion. Input-register ownership errors occur at a different stage.
