"""Random initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class RandomUniform(Initialiser):
    """Initialiser that generates arrays from a uniform distribution."""

    identifier = 'random_uniform'

    def __init__(self, low: float = -0.05, high: float = 0.05) -> None:
        """Initialiser that generates arrays from a uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the range of random values to generate (inclusive).
        high : float, optional
            Upper bound of the range of random values to generate (exclusive).
        """
        self.low = low
        self.high = high

    def call(self, shape: tuple) -> NDArray:
        return np.random.uniform(self.low, self.high, shape)

class RandomNormal(Initialiser):
    """Initialiser that generates arrays from a normal distribution."""

    identifier = 'random_normal'

    def __init__(self, mean: float = 0, stddev: float = 0.05) -> None:
        """Initialiser that generates arrays from a normal distribution.

        Parameters
        ----------
        mean : float, optional
            Mean of the random values to generate.
        stddev : float, optional
            Standard deviation of the random values to generate.
        """
        self.mean = mean
        self.stddev = stddev

    def call(self, shape: tuple) -> NDArray:
        return np.random.normal(self.mean, self.stddev, shape)

