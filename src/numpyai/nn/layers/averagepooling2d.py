"""2D average pooling layer class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.optimisers import Optimiser
from typing import Optional
from .layer import Layer

class AveragePooling2D(Layer):
    """A layer that performs the average pooling operation for 2D spatial data. 
    
    It downsamples the input along its height and width by taking the average value 
    over an input window (of size defined by `pool_size`) for each channel of the input. 
    The window is shifted by `strides` along each dimension.
    """

    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: Optional[tuple[int, int]] = None) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self._called = False

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.input_shape = input_shape

        # Calculates the amount of output rows and columns after pooling
        output_rows = (input_shape[0] - self.pool_size[0]) // self.strides[0] + 1
        output_cols = (input_shape[1] - self.pool_size[1]) // self.strides[1] + 1

        # Stores the output shape and the shape of the view for pooling
        self.output_shape = (output_rows, output_cols) + input_shape[2:]
        self._view_shape = self.output_shape[:2] + self.pool_size + input_shape[2:]
        
        self._built = True
        return self.output_shape

    def call(self, inputs: NDArray, **kwargs) -> NDArray:
        # Builds the layer if it has not yet been built
        if not self._built:
            self.build(inputs.shape[1:])

        # Calculates the strides to be used for creating the input view
        s0, s1 = inputs.strides[1:3]
        pool_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        strides_shape = inputs.strides[:1] + pool_strides + inputs.strides[3:]

        # Creates a view containing the pools of the input
        pools = np.lib.stride_tricks.as_strided(
            inputs, inputs.shape[:1] + self._view_shape, strides=strides_shape
        )

        # Calculates the averages and the average mask for the backwards pass
        averages = np.mean(pools, axis=(3, 4))
        self._average_mask = np.full(pools.shape, 1 / (self.pool_size[0] * self.pool_size[1]))
        self._called = True
        return averages
    
    def backward(self, derivatives: NDArray, _: Optimiser) -> NDArray:
        # Verifies that the layer has been called
        if not self._called:
            raise RuntimeError(
                "Cannot perform a backward pass on "
                "a layer that hasn't been called yet."
            )
        
        # Scales the average mask by the derivatives
        derivatives = derivatives[:, :, :, None, None] * self._average_mask

        # Combines the pools back into the input shape to create delta
        rows, cols = derivatives.shape[1:3]
        delta = np.zeros(derivatives.shape[:1] + self.input_shape)
        for row in range(rows):
            for col in range(cols):
                my, mx = row * self.strides[0], col * self.strides[1]
                ny, nx = my + self.pool_size[0], mx + self.pool_size[1]
                delta[:, my:ny, mx:nx] += derivatives[:, row, col]

        return delta
