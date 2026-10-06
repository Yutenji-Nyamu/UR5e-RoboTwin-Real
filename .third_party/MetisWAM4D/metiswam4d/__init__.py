"""MetisWAM4D: three-expert (Video / Track4D / Action) world-action model.

Modules
-------
schedule      asynchronous noise schedule shared by training and inference
attention     three-expert token layout and visibility rules
experts       expert adapters (local Wan-style DiT, OpenWAM-Alpha wrappers)
focus         coupling field, CSIA and TAA reading interface
model         MetisWAM4D mixture-of-transformers driver
objectives    masked flow-matching and auxiliary losses
sampler       multi-rate asynchronous sampler
data          sample contract, latent shard datasets, mixtures
train         curriculum stages, optimizer groups, FSDP training loop
"""

MODALITIES = ("video", "track", "action")

__all__ = ["MODALITIES"]
