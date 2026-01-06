"""Sigmoid activation function class."""

from numpy.typing import NDArray
from numpyai.nn.backend import sigmoid
from .activation import Activation

class Sigmoid(Activation):
    """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`."""

    identifier = 'sigmoid'

    def call(self, x: NDArray) -> NDArray:
        """Applies the sigmoid function to an input."""
        return sigmoid(x)

    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the sigmoid function to an input."""
        s = sigmoid(x)
        return s * (1 - s)
