"""Human egocentric pretraining data (stage 1): KlingHumanEgo-2.5M-5000H + IG-10K human demonstrations.

Raw 33-frame / stride-4 windows are assembled in dataloader workers (video frames, mu-law Track4D
RGB, condition images, camera codes, prompts) and turned into the model contract on the GPU by
:class:`HumanOnlineEncoder` (Wan VAE + UMT5).  See ``docs/2026-09-24-人类数据预训练-实现记录.md``.
"""
from metiswam4d.data.human.loader import HumanDataConfig, HumanComponentConfig, build_human_loader, collate_human
from metiswam4d.data.human.online_encoder import HumanEncoderConfig, HumanOnlineEncoder

__all__ = [
    "HumanComponentConfig", "HumanDataConfig", "HumanEncoderConfig", "HumanOnlineEncoder", "build_human_loader",
    "collate_human",
]
