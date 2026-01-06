"""Base regulariser class."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpyai.backend import Registrable, Representable

class Regulariser(Representable, Registrable['Regulariser'], ABC):
    """Abstract base class from which all neural network regularisers inherit."""

    identifier: str
    """The regulariser's string identifier."""
    
    def __call__(self, x: NDArray) -> float:
        """Calculates the regularisation penalty for the input."""
        return self.call(x)
    
    @abstractmethod
    def call(self, x: NDArray) -> float:
        """Calculates the regularisation penalty for the input."""

    @abstractmethod
    def derivative(self, x: NDArray) -> NDArray:
        """Calculates the derivative of the regularisation penalty."""
