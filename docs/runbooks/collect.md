# Record and register data

[简体中文](../zh-CN/runbooks/collect.md)

## After each power-up

Clear the workcell, keep a person at the emergency stop, and power/connect the
robot, gripper, and both cameras. Then run:

```bash
conda activate RoboTwinSimReal
cd ~/UR5e_RoboTwin_Real
ur5e-real doctor --config configs/lab.yaml --hardware
```

Start recording only when every item is `OK`. This check reads status only; it
does not move the arm or gripper.

## Record one raw trajectory

Switch PolyScope to Remote Control and confirm that freedrive may be enabled:

```bash
ur5e-real collect --config configs/lab.yaml --task pick_block_bowl \
  --initial-gripper closed
```

`--initial-gripper` records the actual starting state without commanding the
gripper; set it to `open` or `closed` as appropriate. Use
`--note "red block, trial 1"` when a setup variation matters. During capture:

- `c`: close gripper;
- `o`: open gripper;
- `q`: finish normally;
- `Ctrl+C`: finish as interrupted.

Each trajectory creates one `session_<run_id>.json`. It records the task,
start/end/duration, stop mode, code commit, sample/event/frame-pair counts, and
every raw product path. Raw data lives under
`/data/robotics/ur5e-real/raw`.

## Review the result

Use the session path printed when capture finishes:

```bash
ur5e-real review /data/robotics/ur5e-real/raw/action/session_RUN_ID.json \
  --result success --note "clean demonstration"
```

The other results are `failure` and `aborted`. Re-reviewing appends history; it
does not alter image, RTDE, or gripper CSV data. List the latest 20 sessions:

```bash
ur5e-real sessions --config configs/lab.yaml
```

`--save-video` adds preview MP4 files only; synchronized conversion still uses
PNG frames. See [`../DATA_MANAGEMENT.md`](../DATA_MANAGEMENT.md) for retention.
