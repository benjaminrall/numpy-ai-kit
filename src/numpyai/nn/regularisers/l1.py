"""L1 regulariser class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.backend import l1
from .regulariser import Regulariser

class L1(Regulariser):
    """A regulariser that applies an L1 regularisation penalty."""

    identifier = 'l1'

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor

    def call(self, x: NDArray) -> float:
        """Calculates the L1 regularisation penalty for the input."""
        return l1(self.factor, x)

    def derivative(self, x: NDArray) -> NDArray:
        """Calculates the derivative of the L1 regularisation penalty."""
        return self.factor * np.sign(x)
