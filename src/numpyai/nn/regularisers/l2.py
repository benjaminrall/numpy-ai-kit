"""L2 regulariser class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.backend import l2
from .regulariser import Regulariser

class L2(Regulariser):
    """A regulariser that applies an L2 regularisation penalty."""

    identifier = 'l2'

    def __init__(self, factor: float = 0.01) -> None:
        self.factor = factor

    def call(self, x: NDArray) -> float:
        """Calculates the L2 regularisation penalty for the input."""
        return l2(self.factor, x)

    def derivative(self, x: NDArray) -> NDArray:
        """Calculates the derivative of the L2 regularisation penalty."""
        return 2 * self.factor * x
