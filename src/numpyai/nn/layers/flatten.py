"""Flatten layer class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.optimisers import Optimiser
from .layer import Layer

class Flatten(Layer):
    """A layer that flattens the input without affecting the batch size."""

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.input_shape = input_shape
        self.output_shape = (int(np.prod(input_shape)),)
        self._built = True
        return self.output_shape
    
    def call(self, inputs: NDArray, **kwargs) -> NDArray:
        # Builds the layer if it has not yet been built
        if not self._built:
            self.build(inputs.shape[1:])
        return inputs.reshape((inputs.shape[0], -1))
    
    def backward(self, derivatives: NDArray, _: Optimiser) -> NDArray:
        if not self._built:
            raise RuntimeError(
                "Cannot perform a backward pass on "
                "a layer that hasn't been called yet."
            )
        return derivatives.reshape(derivatives.shape[:1] + self.input_shape)