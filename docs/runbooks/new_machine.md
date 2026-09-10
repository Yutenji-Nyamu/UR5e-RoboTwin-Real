# New-machine commissioning and repair

[简体中文](../zh-CN/runbooks/new_machine.md)

This is the bottom-up acceptance checklist. [Setup](setup.md) installs software,
[prerequisites](../PREREQUISITES.md) inventories dependencies,
[hardware commissioning](hardware_commissioning.md) isolates device faults, and
[daily operation](operator_workflows.md) assumes an already accepted workcell.

The repository can rebuild the software and guide device integration; **a clone
alone cannot reproduce the trained robot experiment**. Data, weights and site
configuration are external assets. This checklist was audited against source,
existing physical results and legacy documentation on 2026-09-10; a complete
blank-machine reinstall has not yet been tested.

## 0. Decide what is being replaced

| Scenario | Required work |
| --- | --- |
| New computer, same robot/workcell | Back up experiment assets and pendant installation settings; reinstall, map devices, then rerun each acceptance layer |
| New robot/tool/camera placement/workcell | Recommission and calibrate, establish home, collect fresh joint demonstrations and train a new version; old bounds/weights are not automatically applicable |
| One failing component | Start at that component in section 3, then validate its consumers; do not reinstall the entire Python stack first |

Supported baseline: UR5e e-Series, head/wrist D435i cameras with USB3 data cables,
the current serial gripper and CH340 USB adapter, wired Ethernet, NVIDIA GPU.
The old B81L note about five breakers is site-specific, not a wiring standard.

Keep a local handover record of robot/controller serials and PolyScope version;
IP/subnet/NIC; mounting, workspace and emergency stop; tool mass/center of gravity,
TCP offset and home; gripper model, rated supply and wiring diagram; camera
serial-to-role mapping, mounting photos/orientation, cables and lighting; disk
mount and free capacity. The old notes do **not** supply a complete gripper
procurement/pinout specification or a portable workcell calibration package.
Obtain those from the actual device manual and on-site measurements. CH340 is
the adapter identity, not the gripper model.

## 1. System and Python installation

Prefer Ubuntu 24.04 LTS for a new installation; Ubuntu25.04 is the existing
working site, not the recommended reinstall template.

```bash
sudo apt update
sudo apt install git curl ca-certificates gnupg usbutils build-essential
```

Install the GPU driver following the
[NVIDIA Ubuntu guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html),
reboot and require `nvidia-smi` to work. Choose a driver for the actual GPU/kernel,
not an unconditional downgrade to the old workstation version. Its reported CUDA
version is not proof that Python CUDA packages are installed.
Install Conda and initialize its bash integration using the
[Conda Linux guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html);
open a fresh terminal and check `conda --version`.

Configure the matching distribution repository using the
[RealSense Linux instructions](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md),
then install `librealsense2-dkms`, `librealsense2-utils`, `librealsense2-dev` and
udev rules. Upstream currently lists Ubuntu20/22/24/26 LTS; kernel/DKMS compatibility
still matters. Do not hardcode the old `noble` repository for another release.
Use `sudo usermod -aG dialout "$USER"`, log out/in, verify `id -nG` and reconnect
USB devices. Do not compile the old CH341 driver if the kernel already recognizes
the adapter, or make devices globally writable.

Follow [setup section 1](setup.md) to clone and create `RoboTwinSimReal`.
`environment.yml` includes hardware/DP and the π0.5 client `websockets`/`msgpack`.
It specifies key versions but is not a byte-identical OS/Conda image; record the
resolved packages and driver/kernel versions on the replacement machine.
The π0.5 model uses a separate `.venv/pi05`; do not install JAX into the hardware environment.

## 2. Site configuration, storage and operator commands

Copy `configs/lab.example.yaml` to `configs/lab.yaml` only if no local file exists.

| Setting | How to establish it |
| --- | --- |
| Robot address | Read the pendant; configure a different static computer address in the same wired subnet, without IP conflicts |
| `home_tcp_pose` | Leave null until a clear home is measured in section 3; six values in meters/rotation-vector radians, not Euler angles/degrees |
| Gripper port | `ls -l /dev/serial/by-id/`; stable device path, current protocol at9600baud only for the matching gripper |
| Camera roles | Enumerate, then cover each lens separately; current640×480 RGB,30FPS,10Hz saved frames,60 warmup frames |
| Data root | A real writable directory on the intended mounted disk; do not mix datasets into source control |
| servoJ paths | Keep the example's config-relative XML/script paths, not old-project copies |

See [storage](../STORAGE.md). Do not copy the old `/dev/sda2` or disk UUID.
A new Linux-only disk need not use NTFS; never format an existing data disk as a
setup shortcut. Check `findmnt -T /data/robotics/ur5e-real` and
`df -h /data/robotics/ur5e-real`; an existing directory does not prove the disk is mounted.

```bash
conda activate RoboTwinSimReal
scripts/install_operator_commands.sh
command -v ur5e-collect-init ur5e-pi05-infer
readlink -f /usr/local/bin/ur5e-pi05-infer
ur5e-real doctor --config configs/lab.yaml
```

The installer uses sudo for `/usr/local/bin` symlinks. Entry points are generated
from `pyproject.toml` in the **active hardware Conda environment**, loading this
editable checkout. Verify the active environment before installing. Rerun after
moving the checkout/rebuilding the environment; do not copy old symlinks or an
entire Conda directory. Short commands then work from any directory; the
`python examples/...` commands below still run from this repository in the hardware environment.

## 3. Individual devices: read first, then isolated output

Power/brake release requires an on-site check. After pendant ON/START expect
RUNNING/NORMAL. Select Remote Control for external motion; no competing motion
program should be running. Enable the RTDE service where controller security
settings require it. Read-only software checks do not require playing a URP.
See [ports and programs](../RTDE_STACK.md).

| Layer | Check / pass condition |
| --- | --- |
| Network / UR | `ip -br addr`, `ip route`, then `python examples/smoke/polyscope_status.py --config configs/lab.yaml`; verify mode/state |
| RTDE protocol | `ur5e-real doctor --config configs/lab.yaml --hardware`; expect protocol/controller version, not just an open port |
| Same-packet joint/TCP | `python examples/smoke/rtde_read.py --config configs/lab.yaml --samples 10 --joints`; advancing timestamp, finite q/qd/TCP/offset with correct units |
| Serial | `lsusb`, `ls -l /dev/serial/by-id/`, `id -nG`; doctor proves path existence, not gripper response |
| Camera streams | Inspect individually in `realsense-viewer`, **close viewer**, then `python examples/smoke/realsense.py --config configs/lab.yaml --frames 3`; two640×480 color frames |
| Camera roles / lighting | Cover each lens to verify head/wrist, check color after warmup, USB3 stability, and no competing capture process |

Resolve failures with [hardware troubleshooting](hardware_commissioning.md)
before progressing. The following commands change hardware state: run and
observe **one at a time**, not as a pasted sequence.

```bash
# Arm stationary; gripper area clear. Observe each operation separately.
python examples/smoke/gripper.py open --config configs/lab.yaml --execute
python examples/smoke/gripper.py close --config configs/lab.yaml --execute

# Support the arm; do not issue move commands during freedrive.
python examples/smoke/freedrive.py start --config configs/lab.yaml --execute
python examples/smoke/freedrive.py stop --config configs/lab.yaml --execute
```

Calibrate tool/TCP/payload in PolyScope and enter the measured safe home.
`ur5e-real prepare --config configs/lab.yaml` prints only; add `--execute` after
checking the return path. For an isolated moveL test use
`examples/smoke/socket_move.py --help` with a locally measured target, never
another workcell's six numbers. Gripper completion needs visual confirmation;
this driver has no measured width/force feedback.

## 4. Integrated capture, replay and policy

1. `ur5e-collect-init` **moves to TCP home and opens the gripper**. Restore the scene afterward.
2. Record a short `ur5e-collect pick_place_cube --preview --note "new cell commissioning"`;
   use c/o/q, then s/f/a. Inspect `ur5e-real sessions --config configs/lab.yaml`:
   schema v3, q/qd columns, both cameras and gripper events must be present.
3. Preview `ur5e-replay RUN_ID --max-segments 1`; prepare home/scene, then add
   `--execute`. Socket remains the manual default. Separately preview/test
   `ur5e-replay RUN_ID --backend rtde --chunks 1` for the500Hz TCP chain; it is not
   a joint-policy test.
4. Follow [DP training](train.md)/[inference](infer.md) or the π0.5 section below.
   A new workcell starts with five clean joint demonstrations, retaining at
   least1s after final open. Old TCP-only files cannot be relabeled as joint data.

Require the actual RTDE prime/runtime handshake, not just printed chunks.
Capture, replay and policy execution must not compete for the arm.

## 5. π0.5 environment and experiment assets

```bash
python -m pip install --target .venv/bootstrap uv==0.8.22
bash scripts/bootstrap_robotwin.sh
bash scripts/setup_pi05.sh
.venv/pi05/bin/python -c 'import jax; print(jax.devices())'
```

Require a GPU device. `robotwin.lock` pins upstream; the adapter checks managed
compatibility patches. Do not manually modify/reset upstream model code.

| In Git | Must be supplied separately |
| --- | --- |
| Source, tests, lockfiles, selection and instructions | Full dataset `data/`, `meta/`, `ur5e_adapter/`, plus original camera images |
| Experiment summaries | Complete checkpoint `1000/`, including params, assets/norm, contract, verification and training state for resume |
| Example configuration | Real device mapping, tool calibration and camera placement |
| Download code | For retraining, ~12.5GB base parameters and tokenizer/cache assets or working download access; not automatically fully offline |

The successful dataset is `ur5e/pick_place_cube_joint_5_v20260908`; see
[physical record](../plans/pi05/PHYSICAL_RESULT_20260910.md) for its checkpoint.
Short commands find `<data_root>/pi05/lerobot/<dataset_id>` and the repository's
`checkpoints/pi05/.../1000`. For another experiment pass the same `run:step` to
both init and infer.

**Current portability limit:** `ur5e_adapter/*.npz` stores absolute original-image
paths, also used by model warmup. Copying parquet/checkpoint alone is insufficient.
For the same workcell restore the original `/data/robotics/ur5e-real` layout and
its raw image directories. Automatic relocation to another root is not implemented;
do not alter norms/contracts to bypass validation. Run
`integrations/pi05/check_native.py --dataset ...` in the model environment and the
offline loading instructions in [π0.5 usage](../plans/pi05/USAGE.md) before live testing.

After same-workcell migration checks:

```bash
ur5e-pi05-infer-init 20260909_01:1000 --dry-run
ur5e-pi05-infer 20260909_01:1000 --dry-run
ur5e-pi05-infer 20260909_01:1000 --shadow --chunks 1
```

The first two do not touch hardware/load the model; shadow reads hardware without
actuation. Then verify the workcell and remove init's dry-run / use infer's
`--execute`. Home/offset/contract mismatches require calibration or new data,
not wider guards to force acceptance.

## 6. Repair and handover

| Symptom | Check first |
| --- | --- |
| Intermittent protocol negotiation failure |3 fresh handshake attempts,0.5s spacing now handle transient startup failures. A1s no-reply log does not establish incompatibility; persistent faults need wired-link/controller checks. [Details](hardware_commissioning.md#5-intermittent-rtde-startup-negotiation) |
| Open port, no state | RTDE service, recipe fields and controller version; handshake does not verify500Hz streaming or register ownership |
| Input registers in use / prime failure | Competing process/Fieldbus owner; not an invitation to retry motion |
| Printed chunks, no motion | Remote mode, correct robot program/runtime/nonce, watchdog and measured feedback |
| Serial exists, gripper silent | Supply, protocol/baud, wiring, ownership and permissions |
| Missing/stale/tinted camera | USB3 cable/bandwidth, viewer ownership, serial roles, udev, warmup/lighting |
| Missing/stale command | PATH, symlink, active Conda and editable checkout; reinstall commands in the correct environment |
| Data root not writable | Mount and disk health before chmod; do not repair a mounted volume |
| Model images missing / contract mismatch | Complete assets, raw paths and version identity; keep checks enabled |

Record date, Git SHA, device/driver versions, configuration, layer results, demo
IDs, dataset/checkpoint hashes, offline results, operator outcome and fault logs.
`completed` is not an automatic task-success label.

Legacy coverage: `README.md`/`README_dev.md` network, freedrive, gripper, RealSense
installation/single/dual streams, RTDE read/write, home,10Hz capture, HDF5 and
training/inference map to the steps/links above. `Servoj_RTDE_UR5/README.md`'s
Local/URP flow is replaced by automatic URScript; `CH341SER_LINUX/README.md`'s
manual driver is not required for already recognized hardware. TCP and camera
checks remain independent; durable recordings use the collector.
[Migration](../MIGRATION.md) preserves ACT history without claiming the DP/π0.5
acceptance applies to ACT. Do not inherit the old suggestion to delete raw data
after conversion.
