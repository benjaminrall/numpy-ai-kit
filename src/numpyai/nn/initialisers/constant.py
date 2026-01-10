"""Constant initialiser classes."""

import numpy as np
from numpy.typing import NDArray
from .initialiser import Initialiser

class Zeros(Initialiser):
    """Initialiser that generates arrays filled with zeros."""

    identifier = 'zeros'

    def call(self, shape: tuple) -> NDArray:
        return np.zeros(shape)


class Ones(Initialiser):
    """Initialiser that generates arrays filled with ones."""

    identifier = 'ones'

    def call(self, shape: tuple) -> NDArray:
        return np.ones(shape)


class Constant(Initialiser):
    """Initialiser that generates arrays filled with a constant value."""

    identifier = 'constant'

    def __init__(self, value: float = 0) -> None:
        """Initialiser that generates arrays filled with a constant value.

        Parameters
        ----------
        value : float, optional
            Constant value to be used for initialisation, by default 0.
        """
        self.value = value

    def call(self, shape: tuple) -> NDArray:
        return np.full(shape, self.value, dtype=float)
