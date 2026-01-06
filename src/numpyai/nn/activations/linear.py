"""Linear activation function class."""

import numpy as np
from numpy.typing import NDArray
from .activation import Activation

class Linear(Activation):
    """Linear activation function (pass-through)."""

    identifier = 'linear'

    def call(self, x: NDArray) -> NDArray:
        """Applies the linear function to an input."""
        return x

    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the linear function to an input."""
        return np.ones(x.shape)
