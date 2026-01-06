"""Mean absolute error loss function class."""

from numpy.typing import NDArray
from numpyai.nn.backend import mean_absolute_error
from .loss import Loss

class MeanAbsoluteError(Loss):
    """Computes the mean of the absolute error between outputs and targets."""

    identifier = 'mean_absolute_error'
    aliases = ['mae']

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the mean absolute error loss function."""
        return mean_absolute_error(output, target)

    def derivative(self, output: NDArray, target: NDArray) -> NDArray:
        """Calculates the derivative of the mean absolute error loss function."""
        return 2 * (output - target)
