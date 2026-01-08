"""Softmax activation function class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.backend import softmax
from .activation import Activation

class Softmax(Activation):
    """
    Softmax activation function.

    Converts vectors of values to probability distributions.
    The elements of the output vector are in the range [0, 1] and sum to 1. 

    Each vector in the input is handled independently. The softmax of each 
    vector x is computed as `softmax(x) = exp(x) / reduce_sum(exp(x))`.
    """

    identifier = 'softmax'

    def call(self, x: NDArray) -> NDArray:
        """Applies the softmax function to an input."""
        return softmax(x)

    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the softmax function to an input."""
        return np.ones(x.shape)
