"""RMSprop optimiser class."""

from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numpyai.backend.utils import EPSILON
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class RMSprop(Optimiser):
    """Optimiser that implements the RMSprop algorithm."""

    identifier = 'rmsprop'

    def __init__(self, eta: float = 0.001, rho: float = 0.9) -> None:
        self.eta = eta
        self.rho = rho
        self._one_sub_rho = 1 - rho
        self._averages = defaultdict(Optimiser.zero_cache)

    def optimise_gradients(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the RMSprop optimisation algorithm to the given gradients."""
        average = self._averages[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Maintains a moving discounted average of the square of gradients
            average[i] = self.rho * average[i] + self._one_sub_rho * np.square(gradients[i])

            # Divides the gradient by the root of this average
            gradients[i] = -self.eta * gradients[i] / (np.sqrt(average[i]) + EPSILON)

        return gradients
