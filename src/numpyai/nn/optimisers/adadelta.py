"""Adedelta optimiser class."""

from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numpyai.backend.utils import EPSILON
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class Adadelta(Optimiser):
    """Optimiser that implements the Adadelta algorithm."""

    identifier = 'adadelta'

    def __init__(self, eta: float = 1.0, rho: float = 0.95) -> None:
        self.eta = eta
        self.rho = rho
        self._one_sub_rho = 1 - rho
        self._grad_avgs = defaultdict(Optimiser.zero_cache)
        self._delta_avgs = defaultdict(Optimiser.zero_cache)

    def optimise_gradients(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the Adadelta optimisation algorithm to the given gradients."""
        grad_avg = self._grad_avgs[layer]
        delta_avg = self._delta_avgs[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Maintains a moving discounted average of the square of the gradients
            grad_avg[i] = self.rho * grad_avg[i] + self._one_sub_rho * np.square(gradients[i])

            # Calculates the optimised gradients using the Adadelta algorithm
            gradients[i] = -self.eta * np.sqrt(delta_avg[i] + EPSILON) / np.sqrt(grad_avg[i] + EPSILON) * gradients[i]

            # Updates the average delta value for use in the algorithm
            delta_avg[i] = self.rho * delta_avg[i] + self._one_sub_rho * np.square(gradients[i])

        return gradients
