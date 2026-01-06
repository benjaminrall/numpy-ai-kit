"""Base optimiser class."""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Any, Literal
from abc import ABC, abstractmethod
from collections import defaultdict
from numpy.typing import NDArray
from numpyai.backend import Registrable, Representable

if TYPE_CHECKING:
    from ..layers import TrainableLayer

class Optimiser(Representable, Registrable['Optimiser'], ABC):
    """Abstract base class from which all neural network optimisers inherit."""

    identifier: str
    """The optimiser's string identifier."""

    def __call__(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the optimisation algorithm to the given gradients."""
        return self.optimise_gradients(layer, gradients)
    
    @staticmethod
    def zero() -> Literal[0]:
        """Returns 0 for defaultd dictionary initialisation."""
        return 0

    @staticmethod
    def zero_cache() -> defaultdict[Any, Union[float, NDArray]]:
        """Returns a dictionary that defaults to zeros for optimiser caches."""
        return defaultdict(Optimiser.zero)

    @abstractmethod
    def optimise_gradients(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies an optimisation algorithm to the given gradients."""
