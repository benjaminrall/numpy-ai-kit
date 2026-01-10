"""Adam optimiser class."""

from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numpyai.backend.utils import EPSILON
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class Adam(Optimiser):
    """Optimiser that implements the Adam algorithm.

    Adam optimisation is a stochastic gradient descent method that is based 
    on adaptive estimation of first-order and second-order moments. It is 
    computationally efficient, has little memory requirement, invariant to 
    diagonal rescaling of gradients, and is well suited for problems that are 
    large in terms of data/parameters.
    """

    identifier = 'adam'

    def __init__(self, eta: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, bias_correction: bool = True) -> None:
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.bias_correction = bias_correction
        self._one_sub_beta_1 = 1 - beta_1
        self._one_sub_beta_2 = 1 - beta_2
        self._ms = defaultdict(Optimiser._zero_cache)
        self._vs = defaultdict(Optimiser._zero_cache)
        self._iterations = Optimiser._zero_cache()

    def call(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the Adam optimisation algorithm to the given gradients."""
        iteration = self._iterations[layer] = self._iterations[layer] + 1
        m = self._ms[layer]
        v = self._vs[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Calculates the new first and second order moments (M and V)
            corrected_m = m[i] = self.beta_1 * m[i] + self._one_sub_beta_1 * gradients[i]
            corrected_v = v[i] = self.beta_2 * v[i] + self._one_sub_beta_2 * np.square(gradients[i])

            # Calculates the bias-corrected M and V values
            if self.bias_correction:
                corrected_m = m[i] / (1 - np.power(self.beta_1, iteration))
                corrected_v = v[i] / (1 - np.power(self.beta_2, iteration))

            # Applies the adapted learning rate to the gradients
            gradients[i] = -self.eta * corrected_m / (np.sqrt(corrected_v) + EPSILON)

        return gradients