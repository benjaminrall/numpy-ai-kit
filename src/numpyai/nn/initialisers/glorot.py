"""Glorot initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class GlorotUniform(Initialiser):
    """Glorot uniform initialiser, also called the Xavier uniform initialiser.
    
    Draws samples from a uniform distribution within `[-limit, limit]`, 
    where `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number 
    of input units in the weight array and `fan_out` is the number of output units).
    """

    identifier = 'glorot_uniform'
    aliases = ['xavier_uniform']

    def call(self, shape: tuple) -> NDArray:
        limit = np.sqrt(6 / (shape[-2] + shape[-1]))
        return np.random.uniform(-limit, limit, shape)

class GlorotNormal(Initialiser):
    """Glorot normal initialiser, also called the Xavier normal initialiser.
    
    Draws samples from a normal distribution centred on 0 with 
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is 
    the number of input units in the weight array and `fan_out` 
    is the number of output units.
    """

    identifier = 'glorot_normal'
    aliases = ['xavier_normal']

    def call(self, shape: tuple) -> NDArray:
        scale = np.sqrt(2 / (shape[-2] + shape[-1]))
        return np.random.normal(scale=scale, size=shape)

