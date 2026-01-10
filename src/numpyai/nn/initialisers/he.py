"""He initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class HeUniform(Initialiser):
    """He uniform variance scaling initialiser.
    
    Draws samples from a uniform distribution within `[-limit, limit]`, 
    where `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input 
    units in the weight array).
    """

    identifier = 'he_uniform'

    def call(self, shape: tuple) -> NDArray:
        limit = np.sqrt(6 / shape[-2])
        return np.random.uniform(-limit, limit, shape)

class HeNormal(Initialiser):
    """He normal initialiser.
    
    Draws samples from a normal distribution centred on 0 with 
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number 
    of input units in the weight array.
    """

    identifier = 'he_normal'

    def call(self, shape: tuple) -> NDArray:
        scale = np.sqrt(2 / shape[-2])
        return np.random.normal(scale=scale, size=shape)


