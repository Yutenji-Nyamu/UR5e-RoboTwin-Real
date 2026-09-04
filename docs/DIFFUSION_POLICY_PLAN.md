# RoboTwin Diffusion Policy real-robot implementation context

[简体中文](zh-CN/DIFFUSION_POLICY_PLAN.md)

Status: 2026-09-04. Data conversion, native training, offline inference, and
live shadow pass; servoJ execution is next. Commands are in the training and
inference runbooks.

## Goal and boundary

```text
real raw session
  -> canonical HDF5
  -> RoboTwin DP Zarr
  -> training and checkpoint
  -> offline playback evaluation
  -> live shadow inference
  -> RTDE servoJ execution
```

This repository owns capture, data meaning, live observations, and execution.
The pinned RoboTwin checkout owns the DP model, training, and checkpoint
semantics. It lives at `.third_party/RoboTwin` at commit
`210720340637cb4619283b295dde4cdd807c9e66`.

## Confirmed baseline

| Part | Current conclusion |
|---|---|
| Capture | 10 Hz UR5e RTDE, gripper, and two-D435i capture works on hardware |
| Data | Session `20260903_182752` was recorded, reviewed, and saved successfully |
| Replay | All three socket `movel` segments and both gripper events replayed successfully |
| Historical ACT | Proves real image input, model loading, RTDE read/write, and 500 Hz servoJ integration; it consumed only the first ACT chunk action |
| RoboTwin DP | HDF5 `joint_action/vector` to Zarr, Hydra training, checkpoint, and six-action prediction ran end to end |
| Shadow | Two chunks passed with both cameras and read-only RTDE; steady inference was about 0.84 seconds and sent no command |
| servoJ | Client, RTDE recipe, and robot script are migrated; hold/small-step/continuous-path tests remain on the new stack |

One episode is enough to test conversion, batching, forward passes, checkpoint
loading, and one-episode overfitting. It cannot measure generalization or task
success.

## Frozen first-baseline semantics

| Item | First implementation |
|---|---|
| Model timing | Keep upstream horizon `8`, `3` observation steps, `6` action steps, and `10 Hz` |
| Images | Keep the upstream head-only baseline: `head_camera -> head_cam`; retain wrist images without feeding them yet |
| Real state | Absolute TCP `[x,y,z,rx,ry,rz]` plus physical gripper, open=`0`, closed=`1` |
| 14-value adapter | `[tcp(6), dummy_gripper=0, tcp(6), physical_gripper]` |
| Supervision | `action[t] = state[t+1]`; the action is an absolute TCP target, not a delta |
| Rotation | Make equivalent rotation vectors continuous before writing training data |
| Chunk | Execute all six actions in upstream order, then infer again |
| Robot output | Interpolate six 10 Hz targets into 500 Hz RTDE servoJ setpoints below the model boundary |
| Gripper | Decode element 14 with the existing threshold/debounce semantics |

RoboTwin simulation uses the original 14 values as dual-arm joint positions.
The real adapter preserves the shape but explicitly assigns single-arm TCP
meaning. Real training and inference therefore agree, but simulation joint data
cannot be mixed in without another conversion.

The first baseline has no overlapping-chunk fusion and no early replanning.
Those become separate experiments only if latency, tracking, or success-rate
measurements justify them.

## Implementation stages

### 1. Convert one session (complete)

- Add `/joint_action/vector` to canonical HDF5.
- Add `adapters/robotwin_dp` to produce upstream Zarr keys:
  `data/head_camera`, `data/state`, `data/action`, and `meta/episode_ends`.
- Preserve source run IDs, schema version, and converter commit.
- Check images, 14-value state, next-step action, and episode boundaries on the
  current session.

The current session produced 227 transitions. Native RoboTwin
`RobotImageDataset` loaded `(B,8,3,240,320)` images plus `(B,8,14)` state and
action tensors.

### 2. One-episode training smoke test (complete)

- Complete and pin the DP Python dependencies.
- Use `val_ratio=0` for a single episode, which cannot be split into both train
  and validation sets.
- Use a small batch and few epochs to verify falling loss, checkpoint save, and
  reload.
- Plot all six predicted actions against labels on the same trajectory.

The A6000 completed two epochs with batch 8 and three steps per epoch, producing
and reloading two checkpoints. These are smoke overrides; formal training keeps
the original RoboTwin configuration.

### 3. Offline and live shadow inference (complete)

- Run frame-by-frame inference over recorded data and log predictions and
  latency without robot output.
- Feed live RTDE and head-camera observations into the three-step history.
- Log complete six-action chunks without sending setpoints.

Offline prediction/label comparison is saved, and live shadow completed two
full six-action chunks. This checkpoint validates the pipeline, not task quality.

### 4. servoJ and policy execution (in progress)

- Without a model, test hold, a millimetre target, and one 10 Hz target sequence.
- Interpolate 10 Hz targets to 500 Hz while logging prediction, command, and
  measured TCP.
- Connect DP arm motion, then enable the gripper.

Formal inference remains in Remote mode. A socket sends the robot-side servoJ
program once at startup; state feedback and action targets then use RTDE. This
does not reuse the socket `movel` replay path.

### 5. Formal data and evaluation

- Fix task start conditions, success criteria, and allowed scene variation.
- Collect a small consistent batch, run full training/evaluation, then decide
  whether more demonstrations are useful.
- Associate every checkpoint with dataset ID, config, seed, upstream commit, and
  this repository commit.

## Implemented code entry points

```text
src/ur5e_real/adapters/robotwin_dp/   # HDF5/Zarr, model and semantic adapters
src/ur5e_real/control/                # chunk timing and 500 Hz execution
ur5e-real process-dp                  # HDF5 -> Zarr
ur5e-real train-dp                    # invoke pinned upstream training code
ur5e-real infer-dp                    # offline / shadow / execute
ur5e-infer-init; ur5e-infer           # fixed initialization and live entry point
```

The adapter remains thin: model code is neither copied nor maintained here, and
the ignored upstream checkout is not edited in place.

## Decisions that do not block current work

Offline work can proceed with the existing `pick_place_cube` data and the
baseline above. Later decisions are:

1. Before bulk capture, define success, object-start variation, and collection
   count.
2. Before mixing simulation data, choose real joint actions or an explicit
   joint/TCP conversion.
3. Consider wrist-camera input only after the head-only baseline works.
4. Consider early replanning or chunk fusion only if complete six-step execution
   shows a measured problem.

The immediate next step is to keep PolyScope in Remote and run
`ur5e-infer 600 --execute --chunks 1` to verify automatic startup, servoJ hold,
and one six-target sequence. Then run a complete episode.
