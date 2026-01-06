"""L1L2 regulariser class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.backend import l1, l2
from .regulariser import Regulariser

class L1L2(Regulariser):
    """A regulariser that applies an both L1 and L2 regularisation penalties."""

    identifier = 'l1l2'

    def __init__(self, l1_factor: float = 0.01, l2_factor: float = 0.01) -> None:
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

    def call(self, x: NDArray) -> float:
        """Calculates the L1L2 regularisation penalty for the input."""
        return l1(self.l1_factor, x) + l2(self.l2_factor, x)

    def derivative(self, x: NDArray) -> NDArray:
        """Calculates the derivative of the L1L2 regularisation penalty."""
        return self.l1_factor * np.sign(x) + 2 * self.l2_factor * x
