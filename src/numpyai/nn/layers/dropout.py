"""Dropout layer class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.optimisers import Optimiser
from .layer import Layer

class Dropout(Layer):
    """A layer that applies dropout regularisation to the input. 
    
    It randomly sets input units to 0 with a frequency of `rate` at each step during training, 
    which helps prevent overfitting. Inputs not set to 0 are scaled up by `1 / (1 - rate)` 
    such that the sum over all inputs is unchanged.
    """

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
    
    def call(self, inputs: NDArray, training: bool = False, **kwargs) -> NDArray:
        """Calculates the output of the layer for a given input.

        Parameters
        ----------
        inputs : NDArray
            Input array with any shape.
        training : bool, optional
            Whether to call the layer in training or inference mode. 
            Dropout is only applied when this is True, so that no 
            values are dropped during inference.

        Returns
        -------
        NDArray
            Outputs of the layer with the same shape as the input.
        """

        # Builds the layer if it has not yet been built
        if not self._built:
            self.build(inputs.shape[1:])
        
        # Returns input unchanged if not currently training
        if not training:
            return inputs
        
        # Generates and stores the mask and scale for this pass
        self._mask = np.random.binomial(1, self._inverse_rate, inputs.shape)
        self._called = True

        # Applies the mask and scaling factor to the input
        return inputs * self._mask * self._scale
    
    def backward(self, derivatives: NDArray, _: Optimiser) -> NDArray:
        # Verifies that the layer has been called in training mode
        if not self._called:
            raise RuntimeError(
                "Cannot perform a backward pass on "
                "a Dropout layer that hasn't been called yet."
            )
        return derivatives * self._mask * self._scale
