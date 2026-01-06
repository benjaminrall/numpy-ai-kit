"""Dropout layer class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.optimisers import Optimiser
from .layer import Layer

class Dropout(Layer):
    """A neural network layer that applies dropout to the inputs."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate
        self._inverse_rate = 1 - self.rate
        self._scale = 1 / self._inverse_rate
        self._called = False

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.input_shape, self.output_shape = input_shape, input_shape
        self._built = True
        return input_shape
    
    def call(self, input: NDArray, training: bool = False, **kwargs) -> NDArray:
        # Builds the layer if it has not yet been built
        if not self._built:
            self.build(input.shape[1:])
        
        # Returns input unchanged if not currently training
        if not training:
            return input
        
        # Generates and stores the mask and scale for this pass
        self._mask = np.random.binomial(1, self._inverse_rate, input.shape)
        self._called = True

        # Applies the mask and scaling factor to the input
        return input * self._mask * self._scale
    
    def backward(self, derivatives: NDArray, _: Optimiser) -> NDArray:
        # Verifies that the layer has been called in training mode
        if not self._called:
            raise RuntimeError(
                "Cannot perform a backward pass on "
                "a layer that hasn't been called yet."
            )
        return derivatives * self._mask * self._scale
