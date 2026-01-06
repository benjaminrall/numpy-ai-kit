"""Root mean squared error metric class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.backend import mean_squared_error
from .metric import Metric

class RootMeanSquaredError(Metric):
    """Computes the root of the mean of the squared error between outputs and targets."""

    identifier = 'root_mean_squared_error'
    aliases = ['rmse']

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the root mean squared error."""
        return np.sqrt(mean_squared_error(output, target))