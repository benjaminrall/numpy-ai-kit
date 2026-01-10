"""Adamax optimiser class."""

from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numpyai.backend.utils import EPSILON
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class Adamax(Optimiser):
    """Optimiser that implements the Adamax algorithm.

    Adamax, a variant of Adam based on the infinity norm, is a first-order 
    gradient-based optimisation method. Due to its capabilitiy of adjusting 
    the learning rate based on data characteristics, it is suited to learn 
    time-variant process, e.g. speech data with dynamically changed noise 
    conditions.
    """

    identifier = 'adamax'

    def __init__(self, eta: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self._one_sub_beta_1 = 1 - beta_1
        self._ms = defaultdict(Optimiser._zero_cache)
        self._us = defaultdict(Optimiser._zero_cache)
        self._iterations = Optimiser._zero_cache()

    def call(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the Adamax optimisation algorithm to the given gradients."""
        iteration = self._iterations[layer] = self._iterations[layer] + 1
        m = self._ms[layer]
        u = self._us[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Calculates new M and U values
            m[i] = self.beta_1 * m[i] + self._one_sub_beta_1 * gradients[i]
            u[i] = np.maximum(self.beta_2 * u[i], np.abs(gradients[i]))

            # Adjusts learning rate and applies gradient changes
            current_eta = self.eta / (1 - np.power(self.beta_1, iteration))
            gradients[i] = -current_eta * m[i] / (u[i] + EPSILON)

        return gradients
