"""YAML -> dataclass configuration for a training stage."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from metiswam4d.data.contract import SampleSpec
from metiswam4d.data.human.ig10k import IG10KConfig
from metiswam4d.data.human.kling import KlingConfig
from metiswam4d.data.human.loader import HumanComponentConfig, HumanDataConfig
from metiswam4d.data.human.online_encoder import HumanEncoderConfig
from metiswam4d.data.robodojo.episode_dataset import RDJWindowConfig
from metiswam4d.data.rt2.episode_dataset import RT2WindowConfig
from metiswam4d.data.rt2.online_encoder import RT2EncoderConfig
from metiswam4d.focus import FocusConfig
from metiswam4d.objectives import DropoutConfig, LossWeights, TrackRegionWeights


@dataclass
class ExpertSize:
    dim: int
    ffn_dim: int
    layers: int
    heads: int = 24
    head_dim: int = 128
    freq_dim: int = 256


@dataclass
class ModelConfig:
    experts: tuple[str, ...] = ("video", "track", "action")
    action_read: str = "compact"
    video: ExpertSize = field(default_factory=lambda: ExpertSize(3072, 14336, 30))
    track: ExpertSize = field(default_factory=lambda: ExpertSize(1024, 4096, 20))
    action: ExpertSize = field(default_factory=lambda: ExpertSize(1024, 4096, 30))
    latent_channels: int = 48
    patch: tuple[int, int, int] = (1, 2, 2)
    action_dim: int = 80
    proprio_dim: int = 80
    text_dim: int = 4096
    num_embodiments: int = 8               # embodiment token table rows (registry indices, data/embodiments.py)
    focus: FocusConfig | None = field(default_factory=FocusConfig)
    gradient_checkpointing: bool = True
    dense_read_floor: float | None = None  # compact mode: keep the dense future keys visible to Action at >= this logit bias
    read_action_horizon: int = 0           # > 0: ReadActionHead probes the compact tokens for the clean action chunk


@dataclass
class TrackInit:
    mode: str = "interpolate_from_video"   # interpolate_from_video | checkpoint | fresh
    path: str | None = None                # state-dict path for mode=checkpoint


@dataclass
class InitConfig:
    alpha_checkpoint: str | None = None    # OpenWAM-Alpha dir or .safetensors (video/action/proprio)
    load_video_from_alpha: bool = True     # False when the Video expert comes from initialize_from (previous stage)
    load_action_from_alpha: bool = True
    track: TrackInit = field(default_factory=TrackInit)
    initialize_from: str | None = None     # model weights of a previous stage (DCP dir or .pt)
    initialize_from_prefixes: tuple[str, ...] | None = None  # e.g. [track., reader., progress_] to keep Alpha's Video
    initialize_from_exclude: tuple[str, ...] | None = None   # parameters that keep their fresh init, e.g. [reader.saliency.phi.]
    resume_from: str | None = None         # full training state
    start_step: int = 0                    # with initialize_from: continue the curriculum from this step (fresh optimizer)


@dataclass
class ScheduleConfig:
    completion: dict[str, float] = field(default_factory=lambda: {"video": 1.0, "track": 0.75, "action": 0.25})
    shift: dict[str, float] = field(default_factory=lambda: {"video": 5.0, "track": 5.0, "action": 5.0})
    mixture: dict[str, float] = field(default_factory=lambda: {
        "trajectory": 0.40, "ordered_independent": 0.40, "independent": 0.0,
        "clean_fastest": 0.15, "clean_all_but_slowest": 0.05})
    inference_rounds: int = 16


@dataclass
class DataComponent:
    name: str
    weight: float = 1.0
    manifest: str | None = None            # None -> synthetic
    synthetic_length: int = 64
    modalities: tuple[str, ...] = ("video", "track", "action")
    drop_modalities: tuple[str, ...] = ()
    embodiment: str = "human_ego"
    with_roles: bool = True


@dataclass
class DataConfig:
    spec: SampleSpec = field(default_factory=SampleSpec)
    components: list[DataComponent] = field(default_factory=list)
    batch_size: int = 1
    num_workers: int = 4
    validate: bool = True
    # Raw-episode path with online VAE encoding (RoboTwin 2.0); when set, ``components`` is ignored.
    rt2: RT2WindowConfig | None = None
    rt2_encoder: RT2EncoderConfig | None = None
    # Raw RoboDojo (RDJ_MetisWAM4D) windows, same contract and online encoder as RT2.
    robodojo: RDJWindowConfig | None = None
    robodojo_encoder: RT2EncoderConfig | None = None
    # Human egocentric pretraining (KlingHumanEgo-2.5M-5000H + IG-10K human) with online VAE / UMT5 encoding.
    human: HumanDataConfig | None = None
    human_encoder: HumanEncoderConfig | None = None


@dataclass
class GroupLR:
    lr: float
    freeze_steps: int = 0     # lr = 0 before this step (protect the prior)
    ramp_steps: int = 0       # linear ramp to lr after freeze_steps
    weight_decay: float | None = None


@dataclass
class TrainingConfig:
    max_steps: int = 1000
    groups: dict[str, GroupLR] = field(default_factory=lambda: {
        "video": GroupLR(3e-7, freeze_steps=200, ramp_steps=800),
        "track": GroupLR(3e-6),
        "action": GroupLR(1e-6, freeze_steps=200, ramp_steps=800),
        "focus": GroupLR(1.5e-5),
        "proprio": GroupLR(1e-6),
    })
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    lr_schedule: str = "constant"          # constant | cosine
    lr_schedule_start: int = 0             # cosine: decay runs from this (absolute) step to max_steps; 1.0 before it
    min_lr_ratio: float = 0.1
    dtype: str = "bf16"                    # bf16 | fp32
    fsdp: bool = True
    fsdp_granularity: str = "block"        # block | root
    fsdp_hybrid: bool = True               # HSDP: shard inside a node, replicate across nodes (slow TCP links)
    fsdp_reshard_after_forward: bool = True   # False keeps gathered block weights until backward (+~30 GB at 6-way)
    reduce_dtype: str = "bf16"             # gradient reduction dtype: bf16 | fp32
    grad_accumulation: int = 1             # micro-batches per optimizer step (gradient sync only on the last)
    dense_to_compact_steps: int = 0        # compact mode: anneal Action's dense future reading away over N steps
    dense_to_compact_max_bias: float = 12.0
    log_every: int = 10
    checkpoint_every: int = 1000           # steps; ignored when checkpoint_every_minutes > 0
    checkpoint_every_minutes: float = 0.0  # wall-clock checkpointing (decided on rank 0, broadcast)
    keep_last: int = 3
    visualize: bool = False                # at every checkpoint: sample a fixed batch, write PNGs + TensorBoard images
    visualize_samples: int = 2             # samples per rank in the fixed visualisation batch
    visualize_rounds: int = 16             # sampler rounds
    eval_every: int = 0                    # steps between held-out evaluations (human data: IG-10K val split)
    gpu_filler: bool = False               # sleep kernels on a lowest-priority stream fill the GPU-utilisation gaps
    seed: int = 42
    loss: LossWeights = field(default_factory=LossWeights)
    track_regions: TrackRegionWeights = field(default_factory=TrackRegionWeights)
    dropout: DropoutConfig | None = field(default_factory=DropoutConfig)


@dataclass
class StageConfig:
    name: str
    output_dir: str
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitConfig = field(default_factory=InitConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    notes: str = ""


# ----------------------------------------------------------------------------


def _build(cls, value: Any):
    """Recursively instantiate dataclasses from plain dicts (tuples for tuple fields)."""
    if value is None or not is_dataclass(cls):
        return value
    if is_dataclass(value):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(value).__name__}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in value:
            continue
        raw = value[f.name]
        annotation = str(f.type)
        target = _dataclass_for(f.type)
        if target is not None:
            kwargs[f.name] = _build(target, raw)
        elif annotation.startswith("dict[str, GroupLR]"):
            kwargs[f.name] = {k: _build(GroupLR, v) for k, v in raw.items()}
        elif annotation.startswith("list[DataComponent]"):
            kwargs[f.name] = [_build(DataComponent, v) for v in raw]
        elif annotation.startswith("list[HumanComponentConfig]"):
            kwargs[f.name] = [_build(HumanComponentConfig, v) for v in raw]
        elif annotation.startswith("tuple") and isinstance(raw, list):
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


def _dataclass_for(annotation) -> type | None:
    text = str(annotation)
    for candidate in (ExpertSize, ModelConfig, TrackInit, InitConfig, ScheduleConfig, DataConfig, TrainingConfig,
                      SampleSpec, FocusConfig, LossWeights, TrackRegionWeights, DropoutConfig, GroupLR,
                      RT2WindowConfig, RT2EncoderConfig, RDJWindowConfig, HumanDataConfig, HumanEncoderConfig,
                      HumanComponentConfig,
                      KlingConfig, IG10KConfig):
        if text.startswith(candidate.__name__) or text.startswith(f"{candidate.__name__} |") \
                or text == f"metiswam4d.{candidate.__name__}" or annotation is candidate:
            return candidate
    return None


def load_stage_config(path: str | Path, overrides: dict[str, Any] | None = None) -> StageConfig:
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    if raw.get("model", {}).get("focus", "missing") is None:
        pass  # explicit null keeps focus disabled
    return _build(StageConfig, raw)


def config_to_dict(config) -> Any:
    if is_dataclass(config):
        return {f.name: config_to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [config_to_dict(v) for v in config]
    return config


__all__ = [
    "DataComponent", "DataConfig", "ExpertSize", "GroupLR", "InitConfig", "ModelConfig", "ScheduleConfig",
    "StageConfig", "TrackInit", "TrainingConfig", "config_to_dict", "load_stage_config",
]
