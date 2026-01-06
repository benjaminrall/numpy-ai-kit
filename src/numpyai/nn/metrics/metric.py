"""Base metric class."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Optional
from numpyai.backend import Registrable, Representable

class Metric(Representable, Registrable['Metric'], ABC):
    """Abstract base class from which all neural network metrics inherit."""

    identifier: str
    """The metric's string identifier."""

    def __init__(self, display_name: Optional[str] = None) -> None:
        self._display_name = display_name or self.identifier

    @property
    def __name__(self) -> str:
        return f'{self._display_name}'

    def __call__(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the metric for a given output and target."""
        return self.call(output, target)
    
    @abstractmethod
    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the metric for a given output and target."""
