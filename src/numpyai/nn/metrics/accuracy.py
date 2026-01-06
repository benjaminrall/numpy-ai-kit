"""Accuracy metric class."""

import numpy as np
from numpy.typing import NDArray
from .metric import Metric

class Accuracy(Metric):
    """Calculates how often outputs exactly equal the targets."""

    identifier = 'accuracy'

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the accuracy."""
        return np.sum(output == target) / np.prod(output.shape)