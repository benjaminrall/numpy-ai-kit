"""Dense layer class."""

import numpy as np
from numpy.typing import NDArray
from numpyai.nn.activations import Activation
from numpyai.nn.initialisers import Initialiser
from numpyai.nn.regularisers import Regulariser
from numpyai.nn.optimisers import Optimiser
from .trainable_layer import TrainableLayer

class Dense(TrainableLayer):
    """
    Densely-connected neural network layer.
    
    `output = activation(dot(input, weights) + bias)`
    """

    def __init__(self, units: int, 
                 activation: str | Activation = 'linear', 
                 weight_initialiser: str | Initialiser = 'glorot_uniform', 
                 bias_initialiser: str | Initialiser = 'zeros', 
                 weight_regulariser: str | Regulariser | None = None
                 ) -> None:
        super().__init__()
        self.units = units
        self._called = False

        # Gets activation, initialiser, and regulariser objects
        self.activation = Activation.get(activation)
        self.weight_initialiser = Initialiser.get(weight_initialiser)
        self.bias_initialiser = Initialiser.get(bias_initialiser)
        self.weight_regulariser = Regulariser.get(weight_regulariser) if weight_regulariser else None

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.input_shape = input_shape
        self.output_shape = (input_shape[:-1]) + (self.units,)

        # Initialises weights and biases
        self._weights = self.weight_initialiser((self.input_shape[-1], self.units))
        self._biases = self.bias_initialiser((self.units,))
        self._variables = [self._weights, self._biases]

        self.parameters = self.units * (input_shape[-1] + 1)
        self._built = True
        return self.output_shape
    
    def call(self, input: NDArray, **kwargs) -> NDArray:
        # Builds the layer if it has not yet been built
        if not self._built:
            self.build(input.shape[1:])

        # Stores the input and calculates z output
        self._input = input
        self._z = np.dot(input, self._weights) + self._biases
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
        
        # Calculates gradients for the weights and biases
        nabla_w = np.dot(self._input.T, derivatives)
        nabla_b = np.sum(derivatives, axis=0)

        # Applies regularisation to the weight gradients
        if self.weight_regulariser is not None:
            nabla_w += self.weight_regulariser.derivative(self._weights)

        # Applies optimised gradients to weights and biases
        nabla_w, nabla_b = optimiser(self, [nabla_w, nabla_b])
        self._weights += nabla_w
        self._biases += nabla_b

        # Calculates derivatives for the layer's input
        delta = np.dot(derivatives, self._weights.T)
        return delta
        
    def penalty(self) -> float:
        if self._built and self.weight_regulariser is not None:
            return self.weight_regulariser(self._weights)
        return 0
    
    def set_variables(self, variables: list[NDArray]) -> None:
        super().set_variables(variables)
        self._weights, self._biases = self._variables