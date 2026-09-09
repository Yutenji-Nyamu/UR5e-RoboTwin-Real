"""A switch for native image augmentation; no changes to parameters or flow matching.

Import only after add_native_paths() in the isolated model environment.
"""

import dataclasses

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from flax import nnx


class Pi05TrainingModel(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.image_augmentation = config.image_augmentation

    def compute_loss(self, rng, observation, actions, *, train=False):
        # The locked native compute_loss uses `train` only in image preprocessing.
        # Noise, flow times, loss, trainable parameters and model.train() are unchanged.
        return super().compute_loss(rng, observation, actions, train=train and self.image_augmentation)


@dataclasses.dataclass(frozen=True)
class Pi05TrainingConfig(Pi0Config):
    image_augmentation: bool = False

    def create(self, rng):
        return Pi05TrainingModel(self, nnx.Rngs(rng))
