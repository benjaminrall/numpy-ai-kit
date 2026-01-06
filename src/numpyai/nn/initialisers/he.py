"""He initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class HeNormal(Initialiser):
    """He normal initialiser."""

    identifier = 'he_normal'

    def call(self, shape: tuple) -> NDArray:
        scale = np.sqrt(2 / shape[-2])
        return np.random.normal(scale=scale, size=shape)


class HeUniform(Initialiser):
    """He uniform variance scaling initialiser."""

    identifier = 'he_uniform'

    def call(self, shape: tuple) -> NDArray:
        limit = np.sqrt(6 / shape[-2])
        return np.random.uniform(-limit, limit, shape)
