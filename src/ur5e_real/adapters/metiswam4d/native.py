"""Import the user's local source snapshot without importing upstream data launchers."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import sys


def load_source(root):
    root = Path(root).resolve()
    package = root / "metiswam4d"
    if not (package / "model.py").is_file():
        raise FileNotFoundError(f"MetisWAM4D source missing under {root}")
    existing = sys.modules.get("metiswam4d")
    if existing is not None and Path(existing.__file__).resolve().parent != package:
        raise RuntimeError("a different MetisWAM4D source is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def model_config(*, tiny=False):
    from metiswam4d.experts.local import ExpertCoreConfig

    if tiny:
        video = ExpertCoreConfig(dim=48, ffn_dim=96, num_layers=2, num_heads=2, head_dim=8,
                                 text_dim=16, freq_dim=32)
        action = ExpertCoreConfig(dim=32, ffn_dim=64, num_layers=2, num_heads=2, head_dim=8,
                                  text_dim=16, freq_dim=32)
    else:
        video = ExpertCoreConfig(dim=3072, ffn_dim=14336, num_layers=30)
        action = ExpertCoreConfig(dim=1024, ffn_dim=4096, num_layers=30)
    return {"video": asdict(video), "action": asdict(action), "channels": 4 if tiny else 48,
            "tiny": tiny}


def build_model(config):
    from metiswam4d.experts.local import (
        ActionExpert, ActionExpertConfig, ExpertCoreConfig, LatentExpert, LatentExpertConfig,
    )
    from metiswam4d.model import MetisWAM4D, MetisWAM4DConfig

    video = LatentExpert(LatentExpertConfig(core=ExpertCoreConfig(**config["video"]),
                                          in_channels=config["channels"], out_channels=config["channels"]))
    action = ActionExpert(ActionExpertConfig(core=ExpertCoreConfig(**config["action"]), action_dim=80))
    model = MetisWAM4D(video=video, track=None, action=action, config=MetisWAM4DConfig(
        action_read="none", focus=None, text_dim=config["video"]["text_dim"], proprio_dim=80,
        gradient_checkpointing=not config["tiny"],
    ))
    # action_read=none disables FUTURE world reads. Native attention still exposes
    # the clean current RGB frame to Action. No Track/depth/future-video placeholders.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("action.", "proprio_encoder.")))
    return model


def training_precision(model, *, device, dtype):
    """Frozen weights may use BF16; AdamW must accumulate small updates in FP32."""
    model.to(device=device, dtype=dtype)
    model.action.float()
    model.proprio_encoder.float()
    return model
