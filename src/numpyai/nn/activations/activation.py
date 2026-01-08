"""Base activation function class."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpyai.backend import Registrable, Representable

class Activation(Representable, Registrable['Activation'], ABC):
    """Abstract base class from which all neural network activation functions inherit."""

    identifier: str
    """The activation function's string identifier."""

    def __call__(self, x: NDArray) -> NDArray:
        """Applies the activation function to each element of `x`."""
        return self.call(x)
    
    @abstractmethod
    def call(self, x: NDArray) -> NDArray:
        """Applies the activation function to each element of `x`."""

    @abstractmethod
    def derivative(self, x: NDArray) -> NDArray:
        """Applies the derivative of the activation function to each element of `x`."""