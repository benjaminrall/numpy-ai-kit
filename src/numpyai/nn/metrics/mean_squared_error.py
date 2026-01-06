"""Mean squared error metric class."""

from numpy.typing import NDArray
from numpyai.nn.backend import mean_squared_error
from .metric import Metric

class MeanSquaredError(Metric):
    """Computes the mean of the squared error between outputs and targets."""

    identifier = 'mean_squared_error'
    aliases = ['mse']

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the mean squared error."""
        return mean_squared_error(output, target)