# RTDE read/write stack

[简体中文](zh-CN/RTDE_STACK.md)

## Current implementation

RTDE receive and command execution are separate. Python can subscribe to robot
state directly on port `30004`. To execute poses, Python writes RTDE input
registers while a URScript program on the controller reads those registers and
calls `servoJ`. This project sends that program automatically; no manually
loaded URP is required.

```text
ur5e-infer
├─ cameras + DP model: produce a six-action TCP chunk
├─ port 30001: send robot_programs/servoj_control_loop.script
└─ port 30004: UrRtde client writes input registers at 500 Hz
                         ↓
       controller URScript reads targets, solves IK, calls servoJ
                         ↓
                       UR5e
```

| Layer | Repository-owned component | External requirement |
|---|---|---|
| Python protocol | `UrRtde==2.7.12` pinned in `environment.yml` | Conda environment exists |
| State receive | `src/ur5e_real/hardware/rtde.py` | UR5e port `30004` reachable |
| Setpoint stream | `src/ur5e_real/control/servoj.py` | No competing owner of the same input registers |
| RTDE fields | `robot_programs/control_loop_configuration.xml` | Controller supports RTDE |
| Robot loop | `robot_programs/servoj_control_loop.script` | PolyScope in Remote Control and robot runnable |
| Policy adapter | `src/ur5e_real/adapters/robotwin_dp/` | Pinned RoboTwin checkout and checkpoint |

The current path needs neither ROS, the External Control URCap, nor a URP
manually started on the pendant. The old project's Local-mode
`translation_sample_servoj.urp` flow is historical, not a current DP dependency.

## PolyScope and network requirements

1. Enable remote control under **Settings → System → Remote Control**.
2. Select the **Remote Control** profile from the top-right control profile;
   do not leave the controller in Local mode.
3. Run **ON** and **START** and expect `RUNNING` with safety state `NORMAL`.
4. Put the workstation and UR5e on the same wired subnet; keep the actual
   address in untracked `configs/lab.yaml`.
5. Permit `29999` (Dashboard), `30001` (URScript), and `30004` (RTDE).

UR documents a 500 Hz RTDE target rate for e-Series. Remote Control must first
be enabled in settings and then selected in the control profile. See the
[RTDE manual](https://www.universal-robots.com/manuals/EN/HTML/SW10_10/Content/Prod-RTDE/Real_Time_Data_Exchange_RTDE.htm),
[client port overview](https://www.universal-robots.com/articles/ur/interface-communication/remote-control-via-tcpip/),
and [Remote Control setup](https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/prod-usr-man/software/PolyScope/content/hamburger_menu_g5/System_remote_en.htm).

## Short verification sequence

```bash
# Environment, ports, serial device, and both cameras
ur5e-real doctor --config configs/lab.yaml --hardware

# Read only: subscribe to measured TCP; no pendant program needs to be playing
python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10

# Predict only; never sends robot commands
ur5e-infer 20260905_150221:600 --shadow --chunks 10

# Live: automatically sends URScript and starts the RTDE servoJ stream
ur5e-infer-init
ur5e-infer 20260905_150221:600 --execute
```

“RTDE receives but the robot does not move” is not contradictory: receive only
needs the connection, whereas motion also needs Remote Control, the running
robot-side servoJ program, and an input-register stream. Defaults are the
live-verified ten DP denoising steps, 500 Hz stream, `lookahead=0.1`, and
`gain=300`. Use `--backend socket` only for comparison.
