# Environment setup

[简体中文](../zh-CN/runbooks/setup.md)

The repository owns command implementations and dependency versions. A new
machine still needs system drivers, a Conda environment, and site-local
configuration. Ubuntu 24.04 LTS is the recommended clean baseline. The current
workstation runs a verified Ubuntu 25.04 installation, which is outside the LTS
releases covered by RealSense prebuilt packages.

## 1. Install the repository and Python environment

```bash
git clone git@github.com:Yutenji-Nyamu/UR5e-RoboTwin-Real.git
cd UR5e-RoboTwin-Real
conda env create -f environment.yml
conda activate RoboTwinSimReal
python -m pip install --no-deps -e .
scripts/bootstrap_robotwin.sh
cp configs/lab.example.yaml configs/lab.yaml
```

Edit the untracked `configs/lab.yaml` with the robot address, stable gripper
`/dev/serial/by-id/...` path, camera serials, home TCP, and data root. RoboTwin
is checked out at the revision in `robotwin.lock` under ignored
`.third_party/RoboTwin`; do not copy it manually.

To update an existing environment:

```bash
conda env update -n RoboTwinSimReal -f environment.yml --prune
conda activate RoboTwinSimReal
python -m pip install --no-deps -e .
```

## 2. Install the short operator commands

`pyproject.toml` defines commands such as `ur5e-collect`, `ur5e-replay`, and
`ur5e-infer`. The editable install generates entry points inside the active
Conda environment. That is sufficient when the environment is activated.

To use the same commands from any directory without activating Conda, run once:

```bash
scripts/install_operator_commands.sh
```

The script refreshes the editable install and symlinks every entry point into
`/usr/local/bin`. Code still comes from this repository while Python and
dependencies come from `RoboTwinSimReal`. Rerun it after moving the repository
or recreating the environment. The three layers are:

```text
/usr/local/bin/ur5e-*  ->  Conda environment/bin/ur5e-*  ->  repository src/ur5e_real
```

## 3. System and workcell state

- Install RealSense system packages and udev rules, and add the user to
  `dialout`; see [prerequisites](../PREREQUISITES.md).
- Configure `/data` separately when using the shared data disk; it is not part
  of the Python package. See [workstation storage](../STORAGE.md).
- Enable and select Remote Control in PolyScope; see the complete
  [RTDE read/write stack](../RTDE_STACK.md).

Validate the deployment:

```bash
ur5e-real doctor --config configs/lab.yaml
ur5e-real doctor --config configs/lab.yaml --hardware
```

The first command checks software, configuration, and the data directory. Run
the second after powering and connecting the devices to check network, serial,
and both cameras.
