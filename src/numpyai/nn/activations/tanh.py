"""Linear activation function class."""

import numpy as np
from numpy.typing import NDArray
from .activation import Activation

class Tanh(Activation):
    """Hyperbolic tangent activation function, `tanh(x)`."""

    identifier = 'tanh'

    def call(self, x: NDArray) -> NDArray:
        """Applies the hyperbolic tangent function to an input."""
        return np.tanh(x)

    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the hyperbolic tangent function to an input."""
        return 1 - np.square(np.tanh(x))
