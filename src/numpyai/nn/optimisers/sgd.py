"""SGD optimiser class."""

from collections import defaultdict
from numpy.typing import NDArray
from numpyai.nn.layers import TrainableLayer
from .optimiser import Optimiser

class SGD(Optimiser):
    """Optimiser that implements the stochastic gradient descent algorithm with momentum."""

    identifier = 'sgd'
    aliases = ['stochastic_gradient_descent', 'gradient_descent']

    def __init__(self, eta: float = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        self.eta = eta
        self.momentum = momentum
        self.nesterov = nesterov
        self._velocities = defaultdict(Optimiser.zero_cache)

    def optimise_gradients(self, layer: TrainableLayer, gradients: list[NDArray]) -> list[NDArray]:
        """Applies the SGD optimisation algorithm to the given gradients."""
        velocity = self._velocities[layer]

        # Loops through the gradients for each variable in the layer
        for i in range(len(gradients)):
            # Calculates the gradients scaled by the learning rate
            g = gradients[i] = -self.eta * gradients[i]

            # Applies momentum to the gradients
            if self.momentum > 0:
                v = velocity[i] = self.momentum * velocity[i] + g
                gradients[i] = v if not self.nesterov else self.momentum * v + g

        return gradients
