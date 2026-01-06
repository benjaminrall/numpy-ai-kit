"""Random initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class RandomNormal(Initialiser):
    """Initialiser that generates Numpy arrays from a normal distribution."""

    identifier = 'random_normal'

    def __init__(self, mean: float = 0, stddev: float = 0.05) -> None:
        self.mean = mean
        self.stddev = stddev

    def call(self, shape: tuple) -> NDArray:
        return np.random.normal(self.mean, self.stddev, shape)


class RandomUniform(Initialiser):
    """Initialiser that generates Numpy arrays from a uniform distribution."""

    identifier = 'random_uniform'

    def __init__(self, low: float = -0.05, high: float = 0.05) -> None:
        self.low = low
        self.high = high

    def call(self, shape: tuple) -> NDArray:
        return np.random.uniform(self.low, self.high, shape)
