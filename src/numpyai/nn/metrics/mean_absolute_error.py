"""Mean absolute error metric class."""

from numpy.typing import NDArray
from numpyai.nn.backend import mean_absolute_error
from .metric import Metric

class MeanAbsoluteError(Metric):
    """Computes the mean of the absolute error between outputs and targets."""

    identifier = 'mean_absolute_error'
    aliases = ['mae']

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the mean absolute error."""
        return mean_absolute_error(output, target)
