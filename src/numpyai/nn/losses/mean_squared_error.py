"""Mean squared error loss function class."""

from numpy.typing import NDArray
from numpyai.nn.backend import mean_squared_error
from .loss import Loss

class MeanSquaredError(Loss):
    """Computes the mean of the squared error between outputs and targets."""

    identifier = 'mean_squared_error'
    aliases = ['mse']

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the mean squared error loss function."""
        return mean_squared_error(output, target)

    def derivative(self, output: NDArray, target: NDArray) -> NDArray:
        """Calculates the derivative of the mean squared error loss function."""
        return 2 * (output - target)
