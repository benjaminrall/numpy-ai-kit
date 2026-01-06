"""Glorot initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class GlorotNormal(Initialiser):
    """Glorot normal initialiser, also called the Xavier normal initialiser."""

    identifier = 'glorot_normal'
    aliases = ['xavier_normal']

    def call(self, shape: tuple) -> NDArray:
        scale = np.sqrt(2 / (shape[-2] + shape[-1]))
        return np.random.normal(scale=scale, size=shape)


class GlorotUniform(Initialiser):
    """Glorot uniform initialiser, also called the Xavier uniform initialiser."""

    identifier = 'glorot_uniform'
    aliases = ['xavier_uniform']

    def call(self, shape: tuple) -> NDArray:
        limit = np.sqrt(6 / (shape[-2] + shape[-1]))
        return np.random.uniform(-limit, limit, shape)
