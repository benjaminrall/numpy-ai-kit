"""Nadam optimiser class."""

from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numpyai.backend.utils import EPSILON
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class Nadam(Optimiser):
    """Optimiser that implements the Nadam algorithm."""

    identifier = 'nadam'

    def __init__(self, eta: float = 0.001, beta_1: float = 0.9,
                 beta_2: float = 0.999, bias_correction: bool = True) -> None:
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.bias_correction = bias_correction
        self._one_sub_beta_1 = 1 - beta_1
        self._one_sub_beta_2 = 1 - beta_2
        self._ms = defaultdict(Optimiser.zero_cache)
        self._vs = defaultdict(Optimiser.zero_cache)
        self._iterations = Optimiser.zero_cache()

    def optimise_gradients(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the Nadam optimisation algorithm to the given gradients."""
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

            # Applies Nesterov momentum
            corrected_m = self.beta_1 * corrected_m + self._one_sub_beta_1 * gradients[i]

            # Applies the adapted learning rate to the gradients
            gradients[i] = -self.eta * corrected_m / (np.sqrt(corrected_v) + EPSILON)

        return gradients