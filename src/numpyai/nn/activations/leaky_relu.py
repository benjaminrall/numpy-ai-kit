"""Leaky version of the rectified linear unit activation function class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.backend import leaky_relu
from .activation import Activation

class LeakyReLU(Activation):
    """Rectified linear unit activation function."""

    identifier = 'leaky_relu'

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def call(self, x: NDArray) -> NDArray:
        """Applies the leaky ReLU function to an input."""
        return leaky_relu(x, self.alpha)
    
    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the leaky ReLU function to an input."""
        return np.where(x >= 0, 1, self.alpha).astype(np.double)
    