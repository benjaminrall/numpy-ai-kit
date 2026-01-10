"""2D convolutional layer class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.activations import Activation
from numpyai.nn.initialisers import Initialiser
from numpyai.nn.regularisers import Regulariser
from numpyai.nn.optimisers import Optimiser
from numpyai.backend import dilate_2d
from typing import Optional
from .trainable_layer import TrainableLayer

class Conv2D(TrainableLayer):
    """
    A layer that performs spatial convolution over 2D data. It implements the operation
    `output = activation(conv2d(inputs, kernel) + bias)`.

    This layer creates a set of kernels that are convolved with the layer input 
    to produce its outputs.
    """

    def __init__(self, filters: int,
                 kernel_size: tuple[int, int],
                 strides: tuple[int, int] = (1, 1),
                 activation: str | Activation = 'linear',
                 kernel_initialiser: str | Initialiser = 'glorot_uniform',
                 bias_initialiser: str | Initialiser = 'zeros',
                 kernel_regulariser: Optional[str | Regulariser] = None
                 ) -> None:
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self._called = False

        # Gets activation, initialiser, and regulariser objects
        self.activation = Activation.get(activation)
        self.kernel_initialiser = Initialiser.get(kernel_initialiser)
        self.bias_initialiser = Initialiser.get(bias_initialiser)
        self.kernel_regulariser = Regulariser.get(kernel_regulariser) if kernel_regulariser else None

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.input_shape = input_shape

        # Calculates output rows and cols after convolution
        output_rows = (input_shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        output_cols = (input_shape[1] - self.kernel_size[1]) // self.strides[1] + 1

        # Stores the output shape and the shape of the input view for convolution
        self.output_shape = (output_rows, output_cols, self.filters)
        self._view_shape = self.output_shape[:2] + self.kernel_size + input_shape[2:]

        # Initialises kernels and biases
        self._kernels = self.kernel_initialiser((*self.kernel_size, input_shape[-1], self.filters))
        self._biases = self.bias_initialiser((self.filters,))
        self._variables = [self._kernels, self._biases]

        self.parameters = int(np.prod(self._kernels.shape) + np.prod(self._biases.shape))
        self._built = True
        return self.output_shape
    
    def call(self, inputs: NDArray, **kwargs) -> NDArray:
        # Reshapes inputs that don't have channels to have a single channel
        if len(inputs.shape) == 3:
             inputs = np.reshape(inputs, inputs.shape[:3] + (1,))

        # Builds the layer if it has not yet been built
        if not self._built:
            self.build(inputs.shape[1:])

        # Stores the current input tensor
        self._input = inputs

        # Creates a view of the input containing the sub-matrices for convolution
        s0, s1 = inputs.strides[1:3]
        kernel_strides = (self.strides[0] * s0, self.strides[1] * s1, s0, s1)
        self.input_view = np.lib.stride_tricks.as_strided(
            inputs, inputs.shape[:1] + self._view_shape,
            inputs.strides[:1] + kernel_strides + inputs.strides[3:]
        )

        # Calculates the output of the convolution and applies the activation function to it
        self._z = np.tensordot(self.input_view, self._kernels, axes=3) + self._biases
        self._called = True
        return self.activation(self._z)

    def backward(self, derivatives: NDArray, optimiser: Optimiser) -> NDArray:
        # Verifies that the layer has been called
        if not self._called:
            raise RuntimeError(
                "Cannot perform a backward pass on "
                "a layer that hasn't been called yet."
            )
        
        # Applies derivative of the activation function
        derivatives = self.activation.derivative(self._z) * derivatives

        # Dilates the matrix to account for varying stride values
        dilated_derivatives = dilate_2d(derivatives, self.strides)

        # Pads the input derivative in order to calculate delta
        py, px = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        pd = np.pad(dilated_derivatives, ((0, 0), (py, py), (px, px), (0, 0)))

        # Creates a view of the padded derivative containing the sub-matrices for convolution
        output_shape = (pd.shape[0], pd.shape[1] - self.kernel_size[0] + 1, pd.shape[2] - self.kernel_size[1] + 1)
        derivative_view_shape = output_shape + self.kernel_size + pd.shape[3:]
        derivative_view = np.lib.stride_tricks.as_strided(
            pd, derivative_view_shape, pd.strides[:3] + pd.strides[1:]
        )

        # Calculates gradients for the kernels and biases
        nabla_k = np.tensordot(self.input_view, derivatives, axes=((0, 1, 2), (0, 1, 2)))
        nabla_b = np.sum(derivatives, axis=(0, 1, 2))

        # Applies regularisation to the weight gradients
        if self.kernel_regulariser is not None:
            nabla_k += self.kernel_regulariser.derivative(self._kernels)

        # Applies optimised gradients to weights and biases
        nabla_k, nabla_b = optimiser(self, [nabla_k, nabla_b])
        self._kernels += nabla_k
        self._biases += nabla_b

        # Calculates derivatives for the layer's input
        delta = np.tensordot(derivative_view, self._kernels[::-1, ::-1], axes=((3, 4, 5), (0, 1, 3)))

        # Ensures output is scaled to match the input shape
        if delta.shape != self._input.shape:
            full_delta = np.zeros(self._input.shape)
            full_delta[:, :delta.shape[1], :delta.shape[2]] = delta
            delta = full_delta

        return delta
    
    def penalty(self) -> float:
        if self._built and self.kernel_regulariser is not None:
            return self.kernel_regulariser(self._kernels)
        return 0
    
    def set_variables(self, variables: list[NDArray]) -> None:
        super().set_variables(variables)
        self._kernels, self._biases = variables
