# MetisWAM4D

Three-expert (Video / Track4D / Action) world-action model with asynchronous denoising and the
coupling-salient reading interface (CSIA + TAA). Design: `plan.md`, `docs/JanusAct4D-Method-中文.md`;
implementation plan: `docs/2026-09-21-MetisWAM4D-模型与训练课程实现计划.md`.

```
metiswam4d/            package (see the plan doc §2 for the module map)
configs/               stage1_pretrain_human / stage2_midtrain_ig10k / stage3_* / smoke_tiny
scripts/               env.sh, launch_stage.sh, smoke_tiny.sh, download_alpha_foundation.sh, verify_alpha_equivalence.py
tests/                 CPU unit tests on tiny experts (pytest -q tests)
```

## Quick start

```bash
source scripts/env.sh
python -m pytest -q tests                                    # unit tests (~10 s, CPU)
python -m metiswam4d.train.train --config configs/smoke_tiny.yaml   # tiny synthetic end-to-end run
bash scripts/launch_stage.sh configs/stage1_pretrain_human.yaml     # torchrun, all visible GPUs
bash scripts/launch_stage.sh configs/stage2_midtrain_ig10k.yaml 4 0 10.0.0.1   # 4 nodes, rank 0
```

Overrides use dotted keys: `--set training.max_steps=100 data.batch_size=2`.

## Stages

| Stage | Experts | Data | Init | Config |
| --- | --- | --- | --- | --- |
| S1 pretrain | video, track (+ reader self-supervision, gated read-back into Track) | KlingHumanEgo-2.5M-5000H + IG-10K human | Video ← Alpha Foundation; Track ← interpolated Video | `stage1_pretrain_human.yaml` |
| S2 mid-train | video, track, action (compact reading) | IG-10K robot (real + sim) | S1 + Alpha Foundation action | `stage2_midtrain_ig10k.yaml` |
| S3 post-train | three experts | RoboTwin2 / RoboDojo / real UR5e | S2 | `stage3_robotwin2.yaml`, `stage3_robodojo.yaml`, `stage3_real_ur5e.yaml` |
| S3-X extension | three experts, Video decoupled | IG-10K Cross | S2 | `stage3_ig10k_cross.yaml` |

Manifests under `data.components[*].manifest` must be produced by the offline preprocessing
(video → Track4D → Wan VAE latents, UMT5 text cache); the layout is specified in
`metiswam4d/data/contract.py`.

## RoboTwin 2.0 clean data (RT2_MetisWAM4D)

`docs/2026-09-22-RT2_MetisWAM4D-数据集构建方案.md`. Rendering (t4, python3.10 + SAPIEN):

```bash
bash scripts/data_prep/rt2/make_robotwin_overlay.sh          # once
/usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py manifest && ... link
bash scripts/data_prep/rt2/launch_rt2_track4d.sh demo_clean_4d 16   # then demo_randomized_4d
/usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py status --variants demo_clean_4d
/usr/bin/python3.10 scripts/data_prep/rt2/rt2_track4d.py index --variants demo_clean_4d && ... norm --sample-per-task 5
```

Training reads raw episodes and encodes them online with the Wan2.2 VAE (`data.rt2`, `data.rt2_encoder`):

```bash
METIS_PYTHON=/usr/bin/python3.10 bash scripts/launch_stage.sh configs/rt2_direct.yaml 1 0 127.0.0.1
```

## Verifying the Alpha-compatible experts

```bash
source scripts/env.sh
PYTHONPATH="$OPENWAM_VENDOR:$PYTHONPATH" /usr/bin/python3.10 scripts/verify_alpha_equivalence.py \
    --alpha /m2v_intern_v3/danglingwei/ytech_m2v8_hdd_intern_dataset_danglingwei/model_zoos/OpenWAM/OpenWAM-Alpha-Sim-RoboDojo
```
