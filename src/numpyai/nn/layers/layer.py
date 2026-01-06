"""Base layer class."""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpyai.backend import Representable

if TYPE_CHECKING:
    from ..optimisers import Optimiser

class Layer(Representable, ABC):
    """Abstract base class from which all neural network layers inherit."""

    @property
    def trainable(self) -> bool:
        """Whether the layer is trainable."""
        return False
    
    @property
    def built(self) -> bool:
        """Whether the layer has been built."""
        return self._built

    def __init__(self) -> None:
        self.input_shape: tuple[int, ...]
        self.output_shape: tuple[int, ...]
        self._built: bool = False

    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Builds the layer for a given input shape, returning its output shape."""
    
    def __call__(self, input: NDArray, **kwargs) -> NDArray:
        """Calculates the output of the layer for a given input."""
        return self.call(input, **kwargs)
    
    @abstractmethod
    def call(self, input: NDArray, **kwargs) -> NDArray:
        """Calculates the output of the layer for a given input."""

    @abstractmethod
    def backward(self, derivatives: NDArray, optimiser: Optimiser) -> NDArray:
        """Performs a backwards pass through the layer and applies gradient updates if applicable."""
