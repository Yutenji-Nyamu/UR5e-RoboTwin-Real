# Software, hardware, and layered debugging prerequisites

[简体中文](zh-CN/PREREQUISITES.md)

This replaces the scattered setup notes in the historical README with only the
components this project uses, why they exist, and the shortest debug entry
point.

## Current workstation baseline

| Layer | Current configuration | Purpose |
|---|---|---|
| OS | Ubuntu 25.04 | RealSense, USB serial, and UR network host |
| GPU | NVIDIA RTX A6000 48 GB, driver `580.95.05` | DP training and inference |
| Python | Conda `RoboTwinSimReal`, Python `3.10.18` | Shared hardware/policy environment |
| PyTorch | `2.4.1+cu121`, CUDA available | RoboTwin model runtime |
| Data disk | 4 TB Seagate NTFS3, intended mount `/data` | Raw data, conversions, checkpoints, and logs |
| RoboTwin | `.third_party/RoboTwin` at pinned commit `21072034...` | Upstream model and training code |

On 2026-09-04 the 4 TB drive was still visible as `/dev/sda2`, but `data.mount`
was failing. Restore `/data` before DP conversion; do not relocate canonical raw
sessions back onto the system disk.

## One-time real-hardware setup

### UR5e and PolyScope

- Historical B81L notes say all five breakers must be on. In PolyScope run
  **ON** and **START** and expect `RUNNING/NORMAL`.
- The robot and workstation wired interface use one static subnet; actual
  addresses live only in untracked `configs/lab.yaml`.
- Dashboard `29999` supplies status, URScript `30001` serves initialization,
  capture, and manual replay, and RTDE `30004` carries state and servoJ targets.
- Use `Remote Control` for initialization, freedrive, capture, and socket replay.
- Use `Local` with the robot-side RTDE servoJ program running for policy output.

The repository tracks readable `robot_programs/servoj_control_loop.script` and
the RTDE XML. The historical project also contains the previously used
`translation_sample_servoj.urp`; load and confirm it during the next servoJ
session, then decide whether to retain an exported URP as a robot asset here.

### Gripper

- Power the gripper and connect its USB serial adapter.
- The user must belong to `dialout`.
- Configure its stable `/dev/serial/by-id/...` path rather than a changing
  `/dev/ttyUSB0` name; the actual identity lives only in `configs/lab.yaml`.

### RealSense

- Use one head and one wrist D435i; their serial-role mapping lives only in
  `configs/lab.yaml`.
- Prefer rear USB 3.x ports for both devices.
- System packages are `librealsense2-dkms`, `librealsense2-utils`,
  `librealsense2-dev`, and `librealsense2-udev-rules`.
- Python uses `pyrealsense2==2.56.5.9235`; discard 60 startup frames for auto
  exposure and white balance.

Independent camera viewer:

```bash
realsense-viewer
```

## Software setup

### Repository and base environment

```bash
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install -e .
cp configs/lab.example.yaml configs/lab.yaml
scripts/bootstrap_robotwin.sh
```

For an existing environment use
`conda env update -n RoboTwinSimReal -f environment.yml`. The untracked
`configs/lab.yaml` stores robot address, camera serials, gripper path, home pose,
and data root.

### Additional DP dependencies

The environment already has PyTorch, Torchvision, Diffusers, Zarr 2.x,
NumCodecs, OpenCV, HDF5, Einops, and WandB. The pinned upstream DP also requires
these currently missing modules:

- `hydra-core==1.2.0` and `omegaconf` for upstream training configuration;
- `numba` for batched Zarr sequence sampling;
- `dill` for DP checkpoint save/load.

The first offline implementation will put tested exact versions into
`environment.yml` instead of relying on recurring ad-hoc `pip install` commands.

This project does not require ROS or a full RoboTwin copy inside the repository.
The current kernel and librealsense udev setup already recognize the CH340 and
both cameras, so the old copied CH341 driver is unnecessary.

## Short layered debug path

| Layer | Command/action | Pass condition |
|---|---|---|
| Python/config | `ur5e-real doctor --config configs/lab.yaml` | Python modules, XML, and data root are valid |
| Read-only hardware | Add `--hardware` | UR ports, serial device, and both cameras show `[OK]` |
| Camera view | `python examples/smoke/realsense.py --config configs/lab.yaml --frames 3` | Correct identities and normal post-warmup frames |
| RTDE receive | `python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10` | Continuous 10 Hz TCP output |
| Capture/replay | See [operator quick reference](runbooks/operator_workflows.md) | One end-to-end hardware session is already complete |
| Offline DP | conversion -> one batch -> short train -> checkpoint reload | Completes without a robot connection |
| servoJ | Run the URP in Local mode, then hold/small-step/sequence | Measured TCP follows continuously |
| Online DP | Shadow first, then enable execution | Six predictions, latency, and measured trajectory are logged |

Move down only one layer at a time: offline DP does not touch the robot, servoJ
testing does not load a model, and a failed device begins with its small smoke
command.
