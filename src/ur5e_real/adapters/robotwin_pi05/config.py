"""UR configuration for locked native pi05, with an explicit training augmentation switch."""

from __future__ import annotations

from pathlib import Path
import re

from .contract import STATE_LAYOUT
from .dataset import validate_dataset
from .native import add_native_paths

NAME = "pi05_ur5e_joint_action_expert"
BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"


def action_expert_path(path) -> bool:
    parts = tuple(str(part) for part in path)
    return bool(parts) and (
        parts[0] in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
        or (parts[:2] == ("PaliGemma", "llm") and any(part.endswith("_1") for part in parts[2:]))
    )


def freeze_backbones(path, _value):
    return not action_expert_path(path)


def make_config(
    dataset: Path,
    *,
    exp_name="joint5",
    checkpoint_base=Path("checkpoints/pi05"),
    params=BASE_PARAMS,
    steps=1000,
    batch_size=2,
    learning_rate=2.5e-5,
    warmup_steps=100,
    resume=False,
    schedule_steps=None,
    num_workers=0,
    save_interval=500,
    image_augmentation=False,
):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", exp_name):
        raise ValueError("experiment name must be a simple unique identifier")
    schedule_steps = max(steps, 2) if schedule_steps is None else schedule_steps
    if steps < 1 or not 1 <= batch_size <= 128 or schedule_steps < max(steps, 2):
        raise ValueError("require steps>=1, batch size 1..128, and schedule_steps>=max(steps, 2)")
    if not 0 <= warmup_steps < schedule_steps or not 0 <= num_workers <= 8 or save_interval < 1:
        raise ValueError("require warmup below schedule length, workers 0..8, and save_interval>=1")
    if not 0 < learning_rate <= 1e-3:
        raise ValueError("learning rate must be in (0, 1e-3]")
    contract, ready = validate_dataset(dataset)
    if batch_size > ready["transitions"]:
        raise ValueError("batch exceeds the dataset; native drop_last would produce no training batches")
    add_native_paths()
    from .training_model import Pi05TrainingConfig
    from openpi.training.config import AssetsConfig, LeRobotAlohaDataConfig, TrainConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader
    from openpi.transforms import Group, RepackTransform

    return TrainConfig(
        name=NAME,
        project_name="ur5e-pi05",
        exp_name=exp_name,
        model=Pi05TrainingConfig(pi05=True, image_augmentation=image_augmentation),
        weight_loader=CheckpointWeightLoader(str(params)),
        freeze_filter=freeze_backbones,
        data=LeRobotAlohaDataConfig(
            repo_id=contract["dataset_id"],
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            default_prompt=contract["prompt"],
            assets=AssetsConfig(assets_dir=str(dataset.resolve() / "ur5e_adapter" / "assets"), asset_id=STATE_LAYOUT),
            repack_transforms=Group(
                inputs=[
                    RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        lr_schedule=CosineDecaySchedule(
            warmup_steps=warmup_steps, peak_lr=learning_rate, decay_steps=schedule_steps, decay_lr=learning_rate / 10
        ),
        checkpoint_base_dir=str(checkpoint_base.resolve()),
        ema_decay=None,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=steps,
        log_interval=10,
        save_interval=save_interval,
        keep_period=save_interval,
        resume=resume,
        overwrite=False,
        wandb_enabled=False,
        fsdp_devices=1,
        policy_metadata=contract,
    )
