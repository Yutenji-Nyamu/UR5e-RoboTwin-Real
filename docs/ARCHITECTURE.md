# Architecture

Dependency direction is intentionally one-way:

```text
examples / CLI
      |
collection, data, replay, RoboTwin adapters
      |
control
      |
hardware

RoboTwin adapter ----> ignored pinned RoboTwin checkout
core hardware  -X->   RoboTwin
```

The hardware and data layers never import RoboTwin. This keeps device smoke
tests useful when ML dependencies or GPUs are unavailable. Policy adapters may
import upstream policy code only at runtime and only after the pinned checkout
has been bootstrapped.

The raw on-disk schema preserves the proven historical pipeline: RTDE and
gripper CSV files share a run ID with a dual-camera directory and a timestamp
alignment CSV. Conversion produces RoboTwin-style `episode*.hdf5` files. The
left arm and left wrist observations are placeholders because the physical
workcell currently has one arm and two cameras; the mapping is explicit in
`src/ur5e_real/data/schema.py`.
