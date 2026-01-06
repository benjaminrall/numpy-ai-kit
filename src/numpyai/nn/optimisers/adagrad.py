"""Adagrad optimiser class."""

from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from typing import Union
from numpyai.backend.utils import EPSILON
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class Adagrad(Optimiser):
    """Optimiser that implements the Adagrad algorithm."""

    identifier = 'adagrad'

    def __init__(self, eta: float = 0.01, initial_accumulator_value: float = 0.1) -> None:
        self.eta = eta
        self.initial_accumulator_value = initial_accumulator_value
        self._accumulators = defaultdict(self.accumulator_cache)

    def get_initial_accumulator_value(self) -> float:
        """Returns the initial accumulator value for use in the accumulator cache."""
        return self.initial_accumulator_value

    def accumulator_cache(self) -> defaultdict[int, Union[float, NDArray]]:
        """Returns a default dictionary for accumulator caches."""
        return defaultdict(self.get_initial_accumulator_value)

    def optimise_gradients(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the Adagrad optimisation algorithm to the given gradients."""
        accumulator = self._accumulators[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Adds the square of the current gradients to the accumulator
            accumulator[i] = accumulator[i] + np.square(gradients[i])

            # Uses the accumulator value to adjust the learning rate for each parameter
            gradients[i] = -self.eta / np.sqrt(EPSILON + accumulator[i]) * gradients[i]

        return gradients
